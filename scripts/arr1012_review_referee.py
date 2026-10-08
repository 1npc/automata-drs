"""Independent raw-SBN batch parse plus pinned upstream CLF Referee audit.

No incremental automaton state is used for the raw-to-graph parse. Inventories
are shared; CLF projection is explicitly shared and separately mutation-tested.
Thus this is independent semantic-checker agreement, not fully independent
end-to-end conversion verification. Conversion failures are counted, never
silently repaired or removed. Extra SBN DAG constraints are reported separately.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import shlex
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from automata_drs import validate_sbn, check_sbn_state_wellformedness
from automata_drs.sbn import sbn_graph_to_clf
from automata_drs.sbn.graph import SBNGraph, ConceptNode, ConceptEdge, ConstantEdge, BoxEdge, PropositionEdge
from automata_drs.sbn import spec


def parse_raw(text):
    lexer = shlex.shlex(text, posix=False)
    # SBN constants use double quotes. Apostrophes are ordinary characters
    # in concept lemmas (for example the gold possessive concept 's.n.01).
    # shlex's default single-quote handling would falsely reject that surface.
    lexer.quotes = '"'
    lexer.whitespace_split = True
    lexer.commenters = ""
    words = list(lexer)
    concepts, edges, constants, boxes, props = [], [], [], [], []
    box, active, at = 0, None, 0
    while at < len(words):
        word = words[at]
        if spec.is_synset(word):
            active = len(concepts)
            concepts.append(ConceptNode(active, word, box))
            at += 1
            continue
        if word in spec.SEPARATORS:
            if at+1 == len(words) or not spec.is_box_reference(words[at+1]):
                raise ValueError("separator target missing/malformed")
            ref = words[at+1]
            if not ref.startswith("<"):
                raise ValueError("separator requires backward box target")
            box += 1
            if int(ref[1:]):
                boxes.append(BoxEdge(max(0, box-int(ref[1:])), word, box))
            active = None
            at += 2
            continue
        if word not in spec.ATTACHMENT_LABELS or active is None or at+1 == len(words):
            raise ValueError("unknown/incomplete attachment or no source concept")
        target = words[at+1]
        if word == "Proposition":
            if not spec.is_box_reference(target):
                raise ValueError("malformed Proposition box target")
            offset = int(target[1:]) * (1 if target[0]==">" else -1)
            props.append(PropositionEdge(active, box+offset))
        elif spec.is_node_reference(target):
            edges.append(ConceptEdge(active, word, active+int(target)))
        else:
            if not spec.is_constant(target):
                raise ValueError("malformed constant")
            constants.append(ConstantEdge(active, word, target))
        at += 2
    if not concepts:
        raise ValueError("empty concept register")
    graph = SBNGraph(box+1, tuple(concepts), tuple(edges), tuple(constants), tuple(boxes), tuple(props))
    return graph


def local_valid(text):
    parsed = validate_sbn(text)
    if not parsed.valid:
        return False, parsed.reason
    result = check_sbn_state_wellformedness(parsed.state)
    return result.valid, result.reason


def mutation_fixtures():
    return [
        ("positive", "person.n.01", True),
        ("positive-reference", "person.n.01 help.v.01 Agent -1", True),
        ("positive-apostrophe-lemma", "person.n.01 's.n.01 User -1", True),
        ("positive-quoted-apostrophe", 'person.n.01 Name "Alice\'s book"', True),
        ("placeholder", "entity.n.00", True),
        ("placeholder-constant", "person.n.01 Name ?", True),
        ("dangling-forward", "person.n.01 Theme +9", False),
        ("out-of-range", "person.n.01 Agent -9", False),
        ("dag-cycle-extra-SBN-policy", "person.n.01 Theme +1 person.n.01 Theme -1", False),
        ("empty-box", "person.n.01 NEGATION <1", False),
        ("disconnected", "person.n.01 CONTINUATION <0 person.n.01", False),
        ("accessibility-loop", "thing.n.01 Co-Theme +1 NEGATION <1 money.n.01", False),
        ("dangling-proposition", "person.n.01 Proposition >5", False),
        ("unclosed-constant", 'person.n.01 Name "x', False),
        ("missing-role-target", "person.n.01 Agent", False),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="*", type=Path)
    p.add_argument("--referee", required=True, type=Path)
    p.add_argument("--signature", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        p.error("audit evidence already exists")
    module_spec = importlib.util.spec_from_file_location("upstream_review_referee", args.referee)
    upstream = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(upstream)
    signature = upstream.get_signature(str(args.signature))
    counts = Counter()
    examples = []
    def check(text, label, identity):
        counts[(label, "total")] += 1
        valid, local_reason = local_valid(text)
        counts[(label, "automaton_accept")] += int(valid)
        try:
            graph = parse_raw(text)
            projection = sbn_graph_to_clf(graph)
        except (ValueError, IndexError, KeyError) as exc:
            counts[(label, "independent_conversion_failure")] += 1
            if valid:
                counts[(label, "false_accept_conversion")] += 1
                examples.append({"id": identity, "kind": "conversion", "reason": str(exc), "raw": text})
            return
        results = []
        for name, sig in (("structural", {}), ("official_signature", signature)):
            try:
                upstream.check_clf(list(projection.clauses), sig)
                accepted, reason = True, None
            except RuntimeError as exc:
                accepted, reason = False, str(exc)
            counts[(label, name+"_accept")] += int(accepted)
            if valid != accepted:
                counts[(label, name+"_disagree")] += 1
                if valid:
                    counts[(label, "false_accept_"+name)] += 1
                # Retain every false acceptance, including lexical failures.
                # Only false-rejection examples are capped for report size.
                if valid or len(examples) < 200:
                    examples.append({"id": identity, "kind": name, "local_accept": valid,
                                     "upstream_accept": accepted, "local_reason": local_reason,
                                     "upstream_reason": reason, "raw": text, "clf": projection.clauses})
            results.append(accepted)
    sources = {}
    for path in args.paths:
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if not line.strip():
                    continue
                if path.suffix == ".jsonl":
                    row = json.loads(line)
                    check(row["raw_prediction"], row["configuration"], row["id"])
                else:
                    check(line.rstrip("\n").split("\t",1)[1], "gold", str(index+1))
    fixture_outcomes = []
    for name, text, expected in mutation_fixtures():
        observed, reason = local_valid(text)
        fixture_outcomes.append({"family": name, "expected_automaton": expected, "observed": observed, "reason": reason})
        check(text, "fixture:"+name, name)
    # Perturb CLF directly, bypassing ALL SBN parser/projection logic. These
    # controls validate the imported independent checker and its error path.
    clf_controls = {
        "bound-positive": [("b0","REF","x0"),("b0","person",'"n.01"',"x0")],
        "unbound": [("b0","person",'"n.01"',"x0")],
        "scope-cycle": [("b0","NOT","b1"),("b1","NOT","b0"),
                        ("b0","REF","x0"),("b0","person",'"n.01"',"x0"),
                        ("b1","REF","x1"),("b1","person",'"n.01"',"x1")],
        "disconnected-no-relations": [("b0","REF","x0"),("b0","person",'"n.01"',"x0"),
                                      ("b1","REF","x1"),("b1","person",'"n.01"',"x1")],
        "disconnected": [("b0","NOT","b1"),("b2","NOT","b3"),
                         ("b0","REF","x0"),("b0","person",'"n.01"',"x0"),
                         ("b1","REF","x1"),("b1","person",'"n.01"',"x1"),
                         ("b2","REF","x2"),("b2","person",'"n.01"',"x2"),
                         ("b3","REF","x3"),("b3","person",'"n.01"',"x3")],
    }
    clf_results = {}
    for name, clauses in clf_controls.items():
        try:
            upstream.check_clf(clauses, {})
            clf_results[name] = True
        except RuntimeError:
            clf_results[name] = False
    # Upstream connected_boxes returns False rather than raising on an empty
    # relation, and check_clf does not test that return. Preserve/report this
    # real limitation rather than modifying the independent reference checker.
    if clf_results != {"bound-positive": True, "unbound": False, "scope-cycle": False,
                       "disconnected-no-relations": True, "disconnected": False}:
        raise RuntimeError("independent checker control failure")
    report = {"sources": sources,
              "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "sbn_quote_policy": "double-quoted constants; apostrophes in lemmas remain literal",
              "referee_sha256": hashlib.sha256(args.referee.read_bytes()).hexdigest(),
              "signature_sha256": hashlib.sha256(args.signature.read_bytes()).hexdigest(),
              "independence_scope": "independent batch SBN parse + upstream semantic checker; shared lexical inventories and CLF projection",
              "counts": {label: {metric: count for (group,metric),count in counts.items() if group==label}
                         for label in sorted({key[0] for key in counts})},
              "examples": examples, "fixture_outcomes": fixture_outcomes, "clf_controls": clf_results,
              "upstream_limitation": "empty subordination relation: disconnected_boxes returns False but check_clf ignores it; the SBN automaton independently rejects disconnected boxes",
              "false_accept_structural": sum(v for (_,k),v in counts.items() if k in {"false_accept_conversion","false_accept_structural"})}
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2, ensure_ascii=False)
    with args.output.with_suffix(".manifest.json").open("x", encoding="utf-8") as out:
        json.dump({"status": "complete", "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                   "scientific_agreement_passed": report["false_accept_structural"]==0}, out, indent=2)
    print(json.dumps({"audit": str(args.output), "false_accept_structural": report["false_accept_structural"]}), flush=True)


if __name__ == "__main__":
    main()
