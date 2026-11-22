"""Check published inputs and frozen artifacts without rescoring or model calls."""

import gzip
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
VERSION = "scoring-v1-20261122"


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_gzip(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def main():
    release = ROOT / "analysis" / "releases" / VERSION
    accepted = json.loads((release / "MANIFEST.json").read_text())
    publication = json.loads((ROOT / "analysis/PUBLICATION_MANIFEST.json").read_text())
    redactions = {item["file"]: item for item in publication["redactions"]}
    omitted = set(publication["omitted_frozen_files"])
    require(omitted == {str((release / "analysis/scoring_audit_inputs.json.gz").relative_to(ROOT))},
            "Unexpected omitted frozen files")
    require(accepted["version"] == VERSION, "Unexpected scoring version")
    for name, expected in accepted["files_sha256"].items():
        if str((release / name).relative_to(ROOT)) in omitted:
            require(not (release / name).exists(), "Credential-bearing original must stay outside GitHub")
            require(redactions[name]["original_file_sha256"] == expected,
                    "Redaction provenance disagrees with frozen manifest")
            continue
        require(digest(release / name) == expected, f"Frozen file changed: {name}")
    print(f"Frozen release: {len(accepted['files_sha256']) - len(omitted)} file hashes verified")
    print("One original input withheld, with a separately hashed redacted export")

    # The current manuscript may evolve. Its accepted score sources must not drift.
    for name in ["scoring_audit_inputs.json.gz", "verification_prompt_inputs.json.gz",
                 "manual_answer_review.json", "audited_final_scores.json",
                 "node_score_audit.json"]:
        expected = accepted["files_sha256"]["analysis/" + name]
        if "analysis/" + name in redactions:
            expected = redactions["analysis/" + name]["export_file_sha256"]
        require(digest(ROOT / "analysis" / name) == expected,
                f"Current accepted input or score file changed: {name}")

    main_cases = {}
    for relative in ["benchmark/subset", "half-scale/benchmark/traditional",
                     "half-scale/benchmark/node"]:
        folder = ROOT / "experiments" / relative
        provenance = json.loads((folder / "provenance.json").read_text())
        manifest = folder / "manifest.jsonl"
        require(digest(manifest) == provenance["manifest_sha256"],
                f"Benchmark checksum mismatch: {relative}")
        cases = [json.loads(line) for line in manifest.read_text().splitlines()]
        ids = {case["case_id"] for case in cases}
        require(len(cases) == len(ids) == provenance["prompt_count"],
                f"Benchmark count or duplicate case ID: {relative}")
        if relative.startswith("half-scale"):
            main_cases[relative.rsplit("/", 1)[1]] = {
                case["case_id"]: case for case in cases}
        print(f"Benchmark {relative}: {len(cases):,} cases verified")

    inputs = read_gzip(ROOT / "analysis" / "verification_prompt_inputs.json.gz")
    traditional = main_cases["traditional"]
    require(set(traditional) == set(inputs["traditional_prompts"]),
            "Traditional prompt IDs differ between experiment and audit")
    require(all(case["prompt"] == inputs["traditional_prompts"][key]
                for key, case in traditional.items()), "Traditional prompt text differs")
    nodes = {case["case_id"]: case for case in inputs["node_cases"]}
    require(set(nodes) == set(main_cases["node"]), "Node prompt IDs differ")
    for key, case in main_cases["node"].items():
        require(case["prompt"] == nodes[key]["prompt"], f"Node prompt differs: {key}")
        require(case["graph_data"] == nodes[key]["graph_data"], f"Node graph differs: {key}")
    scoring = read_gzip(ROOT / "analysis" / "scoring_audit_inputs.json.gz")
    require({case["case_id"] for case in scoring["cases"]} == set(traditional),
            "Traditional scoring case IDs differ")
    for case in scoring["cases"]:
        source = traditional[case["case_id"]]
        for field in ["graph_data", "query_params", "task", "order", "prompt_style"]:
            require(source[field] == case[field], f"Scoring metadata differs: {case['case_id']}")
    print("All 19,750 main experiment prompts match the saved audit inputs")

    report = (ROOT / "COMP5222_GraphDO_Report.md").read_text()
    images = re.findall(r"!\[[^\]]*\]\((?:<([^>]+)>|([^\s)]+))\)", report)
    require(len(images) == 11, f"Expected 11 report figures, found {len(images)}")
    for angle, plain in images:
        target = Path(angle or plain)
        require(not target.is_absolute(), f"Report image uses a host-specific path: {target}")
        require((ROOT / target).is_file(), f"Missing report image: {target}")
    print("Current report: all 11 relative image links resolve")
    print("Verification passed within the documented export scope. No scores changed or model requests made.")


if __name__ == "__main__":
    main()
