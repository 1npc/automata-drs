"""Create a new immutable execution version after the predecessor is stopped."""
import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("predecessor", type=Path)
    p.add_argument("root", type=Path)
    p.add_argument("patches", type=Path)
    p.add_argument("--predecessor-jobs", required=True)
    a = p.parse_args()
    queued = subprocess.check_output(["squeue", "-h", "-j", a.predecessor_jobs, "-o", "%i %T"], text=True).strip()
    if queued:
        raise RuntimeError("predecessor jobs still active: " + queued)
    a.root.mkdir(exist_ok=False)
    for folder in ("matrix", "cpu", "logs"):
        (a.root/folder).mkdir()
    shutil.copytree(a.predecessor/"code", a.root/"code", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    patches = {}
    for path in sorted(a.patches.rglob("*")):
        if path.is_file():
            rel = path.relative_to(a.patches)
            if rel.parts[0] not in {"scripts", "tests"} or path.suffix not in {".py", ".sh"}:
                raise RuntimeError("unexpected patch file " + str(rel))
            target = a.root/"code"/rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            patches[str(rel)] = digest(target)
    shutil.copytree(a.predecessor/"vendor", a.root/"vendor")
    for name in ("protocol.md", "ADAPTER_SHA256SUMS", "DATA_SHA256SUMS", "runtime-freeze.txt"):
        shutil.copy2(a.predecessor/name, a.root/name)
    locked = {}
    def lock(path):
        locked[str(path)] = digest(path)
    for name in ("SOURCE_SHA256SUMS", "protocol.md", "ADAPTER_SHA256SUMS", "DATA_SHA256SUMS", "jobs.tsv"):
        lock(a.predecessor/name)
    affected, reuse = [], {}
    for i in range(20):
        old = a.predecessor/"matrix"/f"task-{i}"
        new = a.root/"matrix"/f"task-{i}"
        stem_info = {}
        for path in sorted(old.glob("*.jsonl")):
            lock(path)
            lock(Path(str(path)+".manifest.json"))
            data = path.read_bytes()
            lines = data.splitlines(keepends=True)
            tail = 0
            if lines and not lines[-1].endswith(b"\n"):
                tail = len(lines.pop())
            rows = [json.loads(line) for line in lines]
            failed = sum(bool((row.get("constraint") or {}).get("all_masked_rows", 0)) for row in rows)
            stem_info[path.stem] = {"committed_rows": len(rows), "all_masked_rows": failed,
                                    "incomplete_tail_bytes": tail}
        reuse[str(i)] = stem_info
        fully_unchanged = (i < 8 and not any(info["all_masked_rows"] for info in stem_info.values()))
        if i < 8:
            for folder in (old, a.predecessor/"cpu"/f"score-{i}"):
                if (folder/"exit").read_text().strip() != "0":
                    raise RuntimeError("natural predecessor did not complete")
                for f in folder.iterdir():
                    if f.is_file() and f.name not in {"writer.lock"}:
                        lock(f)
        if fully_unchanged:
            new.symlink_to(old, target_is_directory=True)
            (a.root/"cpu"/f"score-{i}").symlink_to(a.predecessor/"cpu"/f"score-{i}", target_is_directory=True)
        else:
            affected.append(i)
            new.mkdir()
            if i < 8:
                for stem in ("W320", "offline", "natural68", "natural78"):
                    if stem_info[stem]["all_masked_rows"]:
                        continue
                    for suffix in (".jsonl", ".jsonl.manifest.json"):
                        (new/(stem+suffix)).symlink_to(old/(stem+suffix))
    audit = a.predecessor/"cpu"/"audit-existing-0"
    if (audit/"exit").read_text().strip() != "0":
        raise RuntimeError("original independent audit incomplete")
    for f in audit.iterdir():
        if f.is_file() and f.name != "writer.lock":
            lock(f)
    (a.root/"cpu"/"audit-existing-0").symlink_to(audit, target_is_directory=True)
    with (a.root/"PREDECESSOR_SHA256SUMS").open("x", encoding="utf-8") as stream:
        stream.writelines(f"{value}  {path}\n" for path, value in sorted(locked.items()))
    with (a.root/"SOURCE_SHA256SUMS").open("x", encoding="utf-8") as stream:
        stream.writelines(f"{digest(path)}  {path.relative_to(a.root/'code')}\n"
                          for path in sorted((a.root/"code").rglob("*")) if path.is_file())
    accounting = subprocess.check_output(["sacct", "-j", a.predecessor_jobs, "-X", "--parsable2",
                                         "--format=JobID,State,ExitCode,Elapsed"], text=True)
    write_json(a.root/"recovery-plan.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "predecessor": str(a.predecessor),
        "array_indices": affected, "predecessor_rows": reuse, "patch_sha256": patches,
        "predecessor_accounting": accounting,
        "failure_policy": "terminate-at-first-all-masked-v1",
        "scientific_definition_unchanged": True,
        "amendment": "Execution defect: all-inf argmax emitted token zero and kept searching. Stop that input at its first empty support, retain the exact failed prefix and identity, and score zero even if the prefix parses. Rerun every affected old row for valid failure-time cost. Reuse unaffected rows with original cost and per-row provenance; regenerate incomplete/unstarted rows. No model, data, budget, denial rule, metric or threshold is changed. Old failed raw outputs and inflated post-failure costs remain in the predecessor and are not pooled into repaired comparisons.",
    })
    print(json.dumps({"root": str(a.root), "array_indices": affected, "reuse": reuse}), flush=True)


if __name__ == "__main__":
    main()
