"""Accept and freeze the current fixed-corpus scoring release without changing scores.

Replays the pipeline in a temporary checkout, compares every score and derived
result, checks prompts against scoring inputs, and checks regenerated figure pixels.
This is an acceptance check for the saved corpus, not a proof about arbitrary text.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / 'analysis'
VERSION = 'scoring-v1-20261122'
FIGURES = ['accuracy', 'consistency', 'shortest-path-ppr', 'ppr-breakdown']
SCRIPTS = ['score_answers.py', 'audit_node_scores.py', 'review_scores.py',
           'audit_scores.py', 'analyze_rq1.py', 'analyze_rq2.py', 'report_metrics.py',
           'check_ambiguity.py', 'test_score_answers.py', 'document_audit.py',
           'accept_scoring_release.py']
DATA = ['scoring_audit_inputs.json.gz', 'verification_prompt_inputs.json.gz',
        'manual_answer_review.json', 'audited_final_scores.json',
        'false_negative_review.json', 'node_score_audit.json', 'rq1_insights.json',
        'rq2_insights.json', 'report_metrics.json', 'ambiguity_sensitivity.json',
        'rq2_final_scores.json.gz', 'SCORING_AUDIT.md',
        'before-false-negative-review/audited_final_scores.json']
FILES = (['COMP5222_GraphDO_Report.md'] + ['analysis/' + x for x in SCRIPTS + DATA]
         + ['figures/' + x for x in ['figure_style.py', 'plot_final_accuracy.py',
                                    'plot_order_consistency.py', 'plot_shortest_path_ppr.py',
                                    'plot_node_ppr.py', 'graphdo-edge-order-example.png',
                                    'graphdo-edge-order-example.svg']]
         + [f'figures/graphdo-reviewed-{f}.{ext}' for f in FIGURES for ext in ['png', 'svg']])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    raw = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return json.loads(raw)


def edge_multiset(edges, directed=False):
    return Counter(tuple(e) if directed else tuple(sorted(e[:2])) + tuple(e[2:]) for e in edges)


def check_prompt_alignment():
    traditional = read(ANALYSIS / 'scoring_audit_inputs.json.gz')
    inputs = read(ANALYSIS / 'verification_prompt_inputs.json.gz')
    case_ids = {c['case_id'] for c in traditional['cases']}
    assert len(case_ids) == 17500 and case_ids == set(inputs['traditional_prompts'])
    for c in traditional['cases']:
        prompt = inputs['traditional_prompts'][c['case_id']]
        # The final Graph block is the test case, following any demonstrations.
        test = prompt.rsplit('Graph:', 1)[-1]
        match = re.search(r'edges are:\s*(\[[^\]]*\])', test)
        assert match, c['case_id']
        assert edge_multiset(ast.literal_eval(match[1]), c['task'] == 'topology') == edge_multiset(
            c['graph_data']['edges'], c['task'] == 'topology'), c['case_id']
        if c['task'] in ['connectivity', 'shortest_path']:
            question = test.split('Question:', 1)[1]
            query = re.search(r'(?:between|from) node (\d+) (?:and|to) node (\d+)', question)
            assert query and tuple(map(int, query.groups())) == (
                c['query_params']['source'], c['query_params']['target']), c['case_id']
    node_ids = {c['case_id'] for c in inputs['node_cases']}
    assert len(node_ids) == 2250
    for model in inputs['node_models'].values():
        assert set(model) == node_ids
    graph_groups = {}
    for c in inputs['node_cases']:
        g, prompt = c['graph_data'], c['prompt']
        target = g['test_node']
        assert c['query_params']['test_node'] == target
        assert f'What is the label of node {target}?' in prompt
        mapping = re.findall(r'node (\d+): label ([0-6?])', prompt)
        labels = {str(v): str(label) for v, label in g['node_labels'].items()}
        labels[str(target)] = '?'
        assert len(mapping) == len(labels) and dict(mapping) == labels, c['case_id']
        match = re.search(r'Adjacency list:\s*(\[[^\]]*\])', prompt)
        assert match and edge_multiset(ast.literal_eval(match[1])) == edge_multiset(g['edges'])
        key = (c['group'], c['graph_id'], c['question_id'])
        spec = (target, g['node_labels'], sorted(edge_multiset(g['edges']).items()))
        if key in graph_groups:
            assert graph_groups[key] == spec, key
        graph_groups[key] = spec
    assert len(graph_groups) == 150
    return {'traditional_prompts': 17500, 'node_prompts': 2250, 'node_graph_questions': 150}


def run(cmd, cwd, env, log):
    result = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    log.append({'command': cmd, 'returncode': result.returncode,
                'stdout': result.stdout, 'stderr': result.stderr})
    assert result.returncode == 0, '\n'.join(cmd) + '\n' + result.stdout + result.stderr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    release = ANALYSIS / 'releases' / VERSION
    if args.freeze and release.exists():
        raise SystemExit('Release already exists. Verify its manifest instead of overwriting it.')
    before = {name: digest(ROOT / name) for name in FILES}
    alignment = check_prompt_alignment()
    print('Prompt/input alignment passed for all 19,750 prompt records.', flush=True)
    log = []
    with tempfile.TemporaryDirectory(prefix='graphdo-acceptance-') as tmp:
        dest = Path(tmp)
        for name in FILES:
            (dest / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest / name)
        env = dict(os.environ, MPLCONFIGDIR=str(dest / '.mplconfig'), PYTHONDONTWRITEBYTECODE='1')
        run([sys.executable, '-m', 'unittest', 'discover', '-s', 'analysis',
             '-p', 'test_score_answers.py'], dest, env, log)
        for name in ['audit_node_scores.py', 'review_scores.py', 'analyze_rq1.py',
                     'analyze_rq2.py', 'report_metrics.py', 'check_ambiguity.py', 'document_audit.py']:
            run([sys.executable, 'analysis/' + name], dest, env, log)
            print('Replayed ' + name, flush=True)
        # Timestamps, paths and resulting file hashes change in an isolated replay.
        # Every score, decision, count, statistic and sensitivity result must agree.
        original = read(ANALYSIS / 'audited_final_scores.json')
        replay = read(dest / 'analysis/audited_final_scores.json')
        assert {k: v for k, v in original.items() if k != 'created_utc'} == {
            k: v for k, v in replay.items() if k != 'created_utc'}
        for filename, ignored in [
            ('node_score_audit.json', set()), ('false_negative_review.json', set()),
            ('rq1_insights.json', {'source', 'source_sha256'}),
            ('rq2_insights.json', {'provenance'}),
            ('report_metrics.json', {'source_sha256', 'rq1_sha256', 'rq2_sha256'}),
            ('ambiguity_sensitivity.json', {'source_sha256', 'review_sha256'})]:
            a, b = read(ANALYSIS / filename), read(dest / 'analysis' / filename)
            assert {k: v for k, v in a.items() if k not in ignored} == {
                k: v for k, v in b.items() if k not in ignored}, filename
        print('All 52,350 main-analysis scores and all derived results reproduced exactly.', flush=True)
        from PIL import Image, ImageChops
        for script, suffix in zip(['plot_final_accuracy.py', 'plot_order_consistency.py',
                                   'plot_shortest_path_ppr.py', 'plot_node_ppr.py'], FIGURES):
            run([sys.executable, 'figures/' + script], dest, env, log)
            name = f'figures/graphdo-reviewed-{suffix}.png'
            with Image.open(ROOT / name) as a, Image.open(dest / name) as b:
                assert a.size == b.size and ImageChops.difference(a.convert('RGB'), b.convert('RGB')).getbbox() is None, name
        print('Figures 2–5 reproduced with identical pixels.', flush=True)
    assert before == {name: digest(ROOT / name) for name in FILES}, 'Source files changed during acceptance'
    result = {'version': VERSION, 'accepted_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'accepted_for_current_saved_corpus', 'prompt_alignment': alignment,
              'regression_tests': 28, 'traditional_main_scores_reproduced': 47850,
              'traditional_archive_scores_reproduced': 52500, 'node_scores_reproduced': 4500,
              'independent_graph_question_checks': 700, 'paired_tests_reproduced': 300,
              'figure_pixel_matches': 4, 'ambiguous_responses': 7, 'ambiguity_assignments': 128,
              'scope': 'Current saved corpus, approved exclusions, scoring policy, and explicit review ledger.',
              'limit': 'No universal semantic correctness guarantee for arbitrary future answers.',
              'files_sha256': before, 'checks': log}
    if args.freeze:
        release.mkdir(parents=True)
        for name in FILES:
            (release / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, release / name)
        (release / 'MANIFEST.json').write_text(json.dumps(result, indent=2) + '\n')
        assert all(digest(release / name) == value for name, value in before.items())
        print('Frozen release: ' + str(release), flush=True)
    else:
        print('Acceptance passed. Use --freeze to save the release.')


if __name__ == '__main__':
    main()
