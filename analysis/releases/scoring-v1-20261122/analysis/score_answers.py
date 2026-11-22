"""Extract explicit final graph answers and validate them against graph truth.

Answer selection uses wording and position, never whether a candidate is correct.
An unsupported format returns None for explicit review by the caller. The
current audit records exceptional saved answers in a separate review ledger,
without falling back to released scores. Only Python's standard library is used.
"""

from functools import lru_cache
import heapq
import re

LABEL = re.compile(r"^\s*(yes|no)\b", re.I)
FINAL_LABEL = re.compile(r"\b(?:(?:final|the) )?answer(?:\s+is)?\s*[:=]?\s*(yes|no)\b", re.I)
ARROWS = re.compile(r"\d+(?:\s*->\s*\d+)+")
LEGACY_CUE = re.compile(r"\b(?:path|order|ordering|sorting|answer)(?:\s+is)?\s*:\s*", re.I)
PATH_CUE = re.compile(r"\b(?:path|order|ordering|sorting|sort|answer)(?:\s+is)?\s*[:=]\s*", re.I)
FINAL_CUE = re.compile(r"\b(?:final\s+answer|the\s+answer|topological\s+(?:sort|sorting|order|ordering)|shortest\s+path|hamiltonian\s+path)(?:\s+is)?\s*[:=]\s*", re.I)
PREFIX_PATH = re.compile(r"^\s*[\[(]?\s*(\d+(?:(?:\s*->\s*|\s*,\s*)\d+)+|\d+(?:[ \t]+\d+)+)\s*[\])]?", re.I)


def normalized(text):
    for old, new in (("**", ""), ("__", ""), ("`", ""), (r"\(", ""),
                     (r"\)", ""), (r"\[", ""), (r"\]", ""),
                     (r"\rightarrow", "->"), (r"\to", "->"), ("→", "->")):
        text = text.replace(old, new)
    return text.strip()


def explicit_label(text):
    final = list(FINAL_LABEL.finditer(text))
    match = final[-1] if final else (LABEL.match(text.splitlines()[-1]) or LABEL.match(text))
    return None if match is None else match[1].lower() == "yes"


def numeric_list(line):
    parts = re.split(r"\s*(?:->|,)\s*|\s+", line.strip(" []."))
    if parts and all(part.isascii() and part.isdecimal() for part in parts):
        return tuple(map(int, parts))
    return None


def legacy_path_answer(line):
    standalone = numeric_list(line)
    if standalone is not None:
        return standalone
    cue = LEGACY_CUE.search(line)
    if cue:
        labeled = numeric_list(line[cue.end():])
        if labeled is not None:
            return labeled
    arrows = ARROWS.findall(line)
    if len(arrows) == 1 and re.search(r"path|order|weight|length|answer|correction", line, re.I):
        return tuple(map(int, re.findall(r"\d+", arrows[0])))
    return None


# A declaration must introduce an answer, not an edge-verification list such as
# "all edges respect this order:". Selection never uses graph correctness.
DECLARATION = re.compile(
    r"\b(?:topological\s+(?:sort(?:ing)?|order(?:ing)?)"
    r"|(?:one|a)\s+valid\s+order(?:ing)?|correct\s+order(?:ing)?|(?:the\s+|final\s+)?answer"
    r"|(?:(?:one|a|the)\s+(?:valid\s+|such\s+)?|shortest\s+|hamiltonian\s+)paths?"
    r"|\bpath(?=\s*:))"
    r"(?:\s+(?:that\s+)?(?:visits?|visiting)\s+(?:every|each|all)(?:\s+(?:node|vertex|nodes|vertices))?(?:\s+exactly)?\s+once)?"
    r"(?:\s+from(?:\s+node)?\s+\d+\s+to(?:\s+node)?\s+\d+)?"
    r"(?:\s+(?:is|are|exists|itself|would\s+be|achieve\s+it))?\s*[:=]?\s*", re.I)
SEQUENCE = re.compile(r"\d+(?:\s*->\s*\d+)+|\d+(?:\s*,\s*\d+)+|\d+(?:[ \t]+\d+)+")
WITHDRAWAL = re.compile(r"\b(?:cannot|can't|could not|unable to)\s+(?:determine|find|provide|solve)\b", re.I)
BINARY_CONCLUSION = re.compile(
    r"(?:^|\n\s*|\b(?:therefore|thus|so|conclusion|answer)(?:\s+is)?\s*[:,]?\s*)"
    r"(yes|no)(?=\s*[,.:!—–-]|\s*$)", re.I)


def answer_text(text):
    text = normalized(text)
    # Math wrappers and spacing do not change the asserted vertex sequence.
    text = re.sub(r"\\+(?:boxed|text|mathrm)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\+[ ,;!]|\\+(?=[\[\]()])", " ", text)
    text = text.replace("$", "").replace("\ufe0f", "").replace("\u20e3", "")
    text = re.sub(r"\bNode\s+(?=\d+\s*->)|(?<=->)\s*Node\s+", " ", text, flags=re.I)
    text = re.sub(r"\d+(?:[ \t]*[-–−][ \t]*\d+){2,}",
                  lambda m: re.sub(r"[-–−]", "->", m[0]), text)
    text = re.sub(r"\s*--\s*\d+\s*-->\s*", " -> ", text)
    return text


def binary_answer(text, task):
    """Read the last explicit graph conclusion, without consulting truth."""
    text = answer_text(text)
    candidates = [(m.start(), m[1].lower() == "yes") for m in FINAL_LABEL.finditer(text)]
    for m in re.finditer(r"\b(yes|no)(?=\s*[,.:!—–-]|\s*$)", text, re.I):
        candidates.append((m.start(), m[1].lower() == "yes"))
    if task == "connectivity":
        patterns = [(r"there is no path (?:from|between)", False),
                    (r"there is a path (?:from|between)", True)]
    elif task == "cycle":
        patterns = [(r"(?:graph (?:has|contains)|there is) (?:a|an? simple) cycle", True),
                    (r"(?:graph (?:has|contains)|there is) no cycle|no cycle exists|graph is (?:a forest|acyclic)", False)]
    else:
        patterns = [(r"(?:no (?:such |Hamiltonian )?path(?: in this graph)?(?: that)? visits? every node exactly once|no Hamiltonian path|no such path exists|no path in (?:this|the) graph that visits every node exactly once)", False)]
    for pattern, value in patterns:
        candidates.extend((m.start(), value) for m in re.finditer(pattern, text, re.I))
    return max(candidates)[1] if candidates else None


def labeled_prefix(text):
    text = text.lstrip(" [{(\"")
    match = SEQUENCE.match(text)
    if not match:
        return None
    tail = text[match.end():]
    if re.match(r"\.\d", tail):
        return None
    # Mixed comma/arrow runs usually describe separate edges, not one path.
    if re.match(r"\s*(?:,|->)\s*\d", tail):
        return None
    return tuple(map(int, re.findall(r"\d+", match[0])))


def extract_path(text):
    text = answer_text(text)
    if "DSML" in text or "<tool_call" in text:
        return None, None
    lines = [line.strip() for line in text.splitlines()
             if line.strip() and line.strip() not in ["[", "]", "(", ")", "text"]]
    candidates = []
    bare_candidates = []
    if lines and labeled_prefix(lines[0]):
        bare_candidates.append((0, -1, labeled_prefix(lines[0])))
    for i, line in enumerate(lines):
        standalone = numeric_list(line.strip("()"))
        previous = lines[i-1] if i else ""
        if standalone and len(standalone) > 1 and not re.search(
                r"candidate|try|consider|nodes are|vertices are|node set", previous, re.I):
            bare_candidates.append((i, -1, standalone))
        # A bare path followed by its weight is an answer format in GraphDO.
        bare = labeled_prefix(line)
        if bare and re.search(r"shortest path length|total (?:weight|cost)", line, re.I):
            bare_candidates.append((i, -1, bare))
        for cue in DECLARATION.finditer(line):
            before = line[max(0, cue.start()-40):cue.start()].lower()
            if re.search(r"(?:candidate|try|consider|invalid|not a valid)\s*$", before):
                continue
            suffix = line[cue.end():]
            if re.search(r"visits? (?:every|each) node", cue[0], re.I) and (
                    suffix.lstrip().startswith("{") or
                    (not suffix.strip() and i+1<len(lines) and lines[i+1].startswith("{"))):
                continue
            direct_edge = bool(re.match(r"(?:of the graph\s+)?(?:is\s+)?the direct edge", suffix, re.I))
            suffix = re.sub(r"^(?:of the graph\s+)?(?:is\s+)?the direct edge\s*:?\s*", "", suffix, flags=re.I)
            path = labeled_prefix(suffix)
            if direct_edge and path and len(path)==3 and suffix.lstrip().startswith("("):
                path = path[:2]
            if path is None and not suffix.strip(" :") and i+1 < len(lines):
                path = labeled_prefix(lines[i+1])
            # A final answer can specify the weight first and then its path.
            if path is None and re.search(r"answer|shortest", cue[0], re.I):
                conclusion = re.split(r"\b(?:alternative|candidate|verification|check)\b", suffix, flags=re.I)[0]
                via = re.search(r"(?:e\.g\.,?|via|is)\s*[\[(]?\s*(\d+(?:\s*->\s*\d+)+)", conclusion, re.I)
                if via:
                    path = tuple(map(int, re.findall(r"\d+", via[1])))
            if path is not None:
                candidates.append((i, cue.start(), path))
        if re.search(r"minimum (?:weight|cost).*\bvia\b", line, re.I):
            via = re.search(r"\bvia\s+(\d+(?:\s*->\s*\d+)+)", line, re.I)
            if via:
                candidates.append((i, via.start(), tuple(map(int, re.findall(r"\d+", via[1])))))
    # A later unlabelled reasoning fragment must not replace a declared answer.
    # A final numeric line, however, is an explicit conclusion in its own right.
    if bare_candidates and bare_candidates[-1][0] == len(lines)-1:
        candidates.append(bare_candidates[-1])
    if not candidates and len({c[2] for c in bare_candidates}) == 1:
        candidates = bare_candidates
    if candidates:
        i, _, path = max(candidates, key=lambda c: (c[0], c[1]))
        if not any(WITHDRAWAL.search(line) for line in lines[i+1:]):
            return path, "declared_answer_sequence"
    # An unlabelled numeric final line is itself an explicit answer.
    final = numeric_list(lines[-1]) if lines else None
    if final is not None and len(final) > 1:
        return final, "standalone_final_sequence"
    # Preserve a single unambiguous arrow sequence in a concluding sentence.
    if lines and not WITHDRAWAL.search(lines[-1]) and not re.search(
            r"check|edge|candidate|alternative|invalid|not.*valid", lines[-1], re.I):
        arrows = {tuple(map(int, re.findall(r"\d+", s))) for s in ARROWS.findall(lines[-1])}
        if len(arrows) == 1 and re.search(r"path|order|weight|length|answer|correction", lines[-1], re.I):
            return next(iter(arrows)), "unique_final_arrow_sequence"
    return None, None


@lru_cache(maxsize=None)
def graph_for(task, n, edges):
    adjacency = [dict() for _ in range(n)]
    for edge in edges:
        u, v = edge[:2]
        weight = edge[2] if task == "shortest_path" else 1
        adjacency[u][v] = weight
        if task != "topology":
            adjacency[v][u] = weight
    return adjacency


def reachable(graph, source, target):
    seen = {source}
    stack = [source]
    while stack:
        node = stack.pop()
        if node == target:
            return True
        for neighbor in graph[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return False


def has_cycle(graph):
    seen = set()
    def visit(node, parent):
        seen.add(node)
        for neighbor in graph[node]:
            if neighbor == parent:
                continue
            if neighbor in seen or visit(neighbor, node):
                return True
        return False
    return any(visit(node, -1) for node in range(len(graph)) if node not in seen)


@lru_cache(maxsize=None)
def hamiltonian(n, edges):
    graph = graph_for("hamilton", n, edges)
    @lru_cache(maxsize=None)
    def visit(node, mask):
        if mask.bit_count() == n:
            return True
        return any(visit(v, mask | (1 << v)) for v in graph[node] if not mask & (1 << v))
    return any(visit(node, 1 << node) for node in range(n))


@lru_cache(maxsize=None)
def minimum_weight(n, edges, source, target):
    graph = graph_for("shortest_path", n, edges)
    best = {source: 0}
    queue = [(0, source)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != best.get(node):
            continue
        if node == target:
            return cost
        for neighbor, weight in graph[node].items():
            new_cost = cost + weight
            if new_cost < best.get(neighbor, float("inf")):
                best[neighbor] = new_cost
                heapq.heappush(queue, (new_cost, neighbor))
    return float("inf")


def validate_path(case, path, allow_reverse=True):
    task, data = case["task"], case["graph_data"]
    n, edges = data["num_nodes"], tuple(map(tuple, data["edges"]))
    vertices = set(path)
    if not path or any(v < 0 or v >= n for v in vertices) or len(vertices) != len(path):
        return False
    if task == "topology":
        positions = {v: i for i, v in enumerate(path)}
        # Isolated vertices absent from the edge-list prompt may be omitted.
        return all(u in positions and v in positions and positions[u] < positions[v]
                   for u, v in edges)
    graph = graph_for(task, n, edges)
    if any(v not in graph[u] for u, v in zip(path, path[1:])):
        return False
    if task == "hamilton":
        return vertices == set(range(n))
    source, target = case["query_params"]["source"], case["query_params"]["target"]
    if (path[0], path[-1]) != (source, target) and not (
            allow_reverse and (path[0], path[-1]) == (target, source)):
        return False
    return sum(graph[u][v] for u, v in zip(path, path[1:])) == minimum_weight(n, edges, source, target)


def adjudicate(case, answer, legacy=False):
    text = normalized(answer)
    if not text:
        return None, "unextractable", None
    task, data = case["task"], case["graph_data"]
    if not legacy and "DSML" in text:
        return False, "tool_call_without_answer", None
    if not legacy and "<tool_call" in text:
        text = text.split("<tool_call")[0].strip()
        answer = text
        if not text:
            return False, "tool_call_without_answer", None
    n, edges = data["num_nodes"], tuple(map(tuple, data["edges"]))
    graph = graph_for(task, n, edges)
    label = explicit_label(text)
    if not legacy and task == "hamilton":
        cleaned = answer_text(text)
        path, _ = extract_path(text)
        path_positions = [m.start() for m in SEQUENCE.finditer(cleaned)
                          if tuple(map(int, re.findall(r"\d+", m[0]))) == path] if path else []
        negative = list(re.finditer(
            r"\b(?:no Hamiltonian path(?! is provided)(?: exists)?"
            r"|no such path(?: exists)?"
            r"|no (?:valid )?path(?: in (?:this|the) graph)? (?:that |can |could )?visit(?:s|ing)? (?:every|all|each) (?:node|vertex|vertices)"
            r"|(?:does not|doesn't) (?:have|contain) (?:a )?Hamiltonian path)", cleaned, re.I))
        negative = [m for m in negative if not cleaned[m.end():].lstrip().startswith("?")]
        if negative and negative[-1].start() > max(path_positions, default=-1):
            label = False
    if not legacy and label is None:
        label = binary_answer(text, task)
    if task in ["cycle", "connectivity"]:
        if label is None:
            return None, "unextractable", None
        expected = has_cycle(graph) if task == "cycle" else reachable(
            graph, case["query_params"]["source"], case["query_params"]["target"])
        return label == expected, "explicit_binary_label", label
    if not legacy and task == "hamilton" and label is None:
        last_paragraph = text.split("\n\n")[-1]
        if re.search(r"\b(?:no Hamiltonian path exists|Hamiltonian path does not exist|there is no (?:Hamiltonian )?path (?:that )?visits? every node exactly once|(?:does not|doesn't) (?:have|contain) (?:a )?Hamiltonian path)\b", last_paragraph, re.I):
            label = False
    if task == "hamilton" and (label is False or re.search(r"\bthere is no hamiltonian path\b", text.splitlines()[-1], re.I)):
        return not hamiltonian(n, edges), "explicit_negative_hamiltonian", False
    if legacy:
        path = legacy_path_answer(text.splitlines()[-1])
        kind = "legacy_final_path"
    else:
        path, kind = extract_path(text)
        if path is None:
            path = legacy_path_answer(text.splitlines()[-1])
            kind = "legacy_final_path"
    if path is None:
        return None, "unextractable", None
    return validate_path(case, path, allow_reverse=not legacy), kind, path
