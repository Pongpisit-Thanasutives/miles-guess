"""Merge explicit source reviews with conservative, per-function body coverage.

Run from any directory after generating validation/audit-coverage.json:
    python examples/build_function_audit.py

Coverage of a definition at import time is never counted as invocation. This
report records execution evidence, not correctness or complete branch testing.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


DEFAULT_LEDGERS = (
    "validation/audit-torch.json",
    "validation/audit-torch-checkpoint.json",
    "validation/audit-keras.json",
    "validation/audit-metrics.json",
    "validation/audit-applications.json",
    "validation/audit-evaluation.json",
)

# These three small expressions were separately source-reviewed on 2026-10-17.
# Exact expression matching prevents silently carrying a review onto changed code.
LAMBDA_REVIEWS = {
    ("mlguess/classifier_uq.py", "lambda c: int(c[9:])"):
        "Reviewed column sorting key: converts the numeric suffix after the fixed nine-character prediction-column prefix.",
    ("mlguess/torch/checkpoint.py", "lambda x: x.detach().cpu() if torch.is_tensor(x) else x"):
        "Reviewed tensor mapping: detaches tensors and moves them to CPU, preserving non-tensor state entries.",
    ("mlguess/torch/distributed.py", "lambda submodule: size_based_auto_wrap_policy(submodule, recurse=False, nonwrapped_numel=sum((p.numel() for p in submodule.parameters())), min_num_params=100000)"):
        "Reviewed activation-checkpoint selection predicate against the native size policy call. Accelerator execution remains unverified.",
}


def span(node):
    return set(range(node.lineno, getattr(node, "end_lineno", node.lineno) + 1))


def function_body_lines(node):
    """Exclude signatures, decorators, docstrings and independently scoped code."""
    own_lines, statement_starts, excluded = set(), set(), {node.lineno}

    def walk(child):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            excluded.update(span(child))
            for decorator in child.decorator_list:
                excluded.update(span(decorator))
            return
        if isinstance(child, ast.Lambda):
            # A lambda's expression line may execute on creation or invocation.
            # Removing the overlapping parent line is deliberately conservative.
            excluded.update(span(child))
            return
        if isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant) and isinstance(child.value.value, str):
            excluded.update(span(child))
            return
        if isinstance(child, ast.stmt):
            statement_starts.add(child.lineno)
            own_lines.update(span(child))
        for descendant in ast.iter_child_nodes(child):
            walk(descendant)

    for statement in node.body:
        walk(statement)
    own_lines.difference_update(excluded)
    return own_lines, statement_starts & own_lines


class Inventory(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename
        self.scope = []
        self.records = []

    def visit_ClassDef(self, node):
        for expression in [*node.decorator_list, *node.bases, *(keyword.value for keyword in node.keywords)]:
            self.visit(expression)
        self.scope.append(("class", node.name))
        # Class decorators/bases are creation-time expressions, not method bodies.
        for statement in node.body:
            self.visit(statement)
        self.scope.pop()

    def visit_FunctionDef(self, node):
        # Lambdas in defaults/decorators belong to the defining scope too.
        for expression in [*node.decorator_list, node.args, node.returns]:
            if expression is not None:
                self.visit(expression)
        parents = [name for _, name in self.scope]
        name = ".".join(parents + [node.name])
        nested = any(kind == "function" for kind, _ in self.scope)
        kind = "nested_function" if nested else ("method" if self.scope else "function")
        body_lines, markers = function_body_lines(node)
        self.records.append({
            "file": self.filename,
            "function": name,
            "kind": "async_" + kind if isinstance(node, ast.AsyncFunctionDef) else kind,
            "line": node.lineno,
            "end_line": node.end_lineno,
            "column": node.col_offset,
            "source_ast_sha256": hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest(),
            "_body_lines": body_lines,
            "_markers": markers,
        })
        self.scope.append(("function", node.name))
        for statement in node.body:
            self.visit(statement)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        name = ".".join([name for _, name in self.scope] + [f"<lambda>@{node.lineno}:{node.col_offset}"])
        self.records.append({
            "file": self.filename,
            "function": name,
            "kind": "lambda",
            "line": node.lineno,
            "end_line": node.end_lineno,
            "column": node.col_offset,
            "source_ast_sha256": hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest(),
            "_expression": ast.unparse(node),
            "_body_lines": set(),
            "_markers": set(),
        })
        self.generic_visit(node)


def relative_name(filename, root):
    path = Path(filename)
    if path.is_absolute():
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix().removeprefix("./")


def load_reviews(root, filenames):
    reviews, metadata, missing = defaultdict(list), [], []
    for filename in filenames:
        path = root / filename
        if not path.exists():
            missing.append(filename)
            continue
        content = json.loads(path.read_text(encoding="utf-8"))
        metadata.append({"file": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for item in content.get("functions", []):
            if not isinstance(item, dict):
                continue
            module = item.get("file", item.get("source", item.get("module", "")))
            function = item.get("function", item.get("name", ""))
            if not module or not function:
                continue
            if "/" not in module and "\\" not in module and not module.endswith(".py"):
                module = module.replace(".", "/") + ".py"
            module = relative_name(module, root)
            function = function.replace(".<locals>.", ".")
            reviews[(module, function)].append(dict(item, ledger=filename))
    return reviews, metadata, missing


def explicit_review(item):
    if "reviewed" in item:
        return item["reviewed"] is True
    if str(item.get("status", "")).lower() in {"review", "reviewed", "source-reviewed"}:
        return True
    note = str(item.get("notes", "")).lower()
    return any(term in note for term in ("source reviewed", "source-reviewed", "read-only scientific review"))


def texts(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [str(value)]


def build(root, coverage_file, ledger_files):
    reviews, ledger_metadata, missing_ledgers = load_reviews(root, ledger_files)
    coverage_path = root / coverage_file
    if not coverage_path.exists():
        raise FileNotFoundError(f"Coverage JSON is required: {coverage_path}")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    covered_files = {relative_name(name, root): value for name, value in coverage.get("files", {}).items()}
    rows, source_files = [], []
    for directory in ("mlguess", "applications"):
        for path in sorted((root / directory).rglob("*.py")):
            if "tests" in path.relative_to(root).parts:
                continue
            filename = path.relative_to(root).as_posix()
            visitor = Inventory(filename)
            visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=filename))
            source_files.append({"file": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "functions": len(visitor.records)})
            file_coverage = covered_files.get(filename)
            for record in visitor.records:
                body_lines = record.pop("_body_lines")
                markers = record.pop("_markers")
                expression = record.pop("_expression", None)
                ledger_entries = reviews.get((filename, record["function"]), [])
                supplemental = LAMBDA_REVIEWS.get((filename, expression))
                record["source_review"] = "reviewed" if any(explicit_review(item) for item in ledger_entries) or supplemental else "not_recorded"
                record["ledger_statuses"] = sorted({str(item.get("status", "")) for item in ledger_entries})
                record["review_ledgers"] = sorted({item["ledger"] for item in ledger_entries})
                record["tests_listed_by_reviewers"] = sorted({str(test) for item in ledger_entries for test in texts(item.get("tests"))})
                record["review_notes"] = list(dict.fromkeys(str(note) for item in ledger_entries for note in texts(item.get("notes"))))
                if supplemental:
                    record["review_notes"].append(supplemental)
                    record["review_ledgers"].append("examples/build_function_audit.py::LAMBDA_REVIEWS (exact expression match)")
                record["reviewer_limitations"] = list(dict.fromkeys(str(note) for item in ledger_entries for key in ("unexecuted_branches", "execution_scope") for note in texts(item.get(key))))
                executed = set((file_coverage or {}).get("executed_lines", []))
                missing = set((file_coverage or {}).get("missing_lines", []))
                excluded = set((file_coverage or {}).get("excluded_lines", []))
                known_body = (executed | missing) & body_lines - excluded
                executed_body = executed & known_body
                missing_body = missing & known_body
                if record["kind"] == "lambda":
                    evidence = "indeterminate_lambda_line"
                elif file_coverage is None:
                    evidence = "no_coverage_data"
                elif executed_body:
                    evidence = "body_execution_observed"
                elif known_body:
                    evidence = "body_execution_not_observed"
                else:
                    evidence = "no_unambiguous_body_marker"
                record["execution_evidence"] = evidence
                record["body_lines_executed"] = sorted(executed_body)
                record["body_lines_missing"] = sorted(missing_body)
                record["body_lines_total"] = len(known_body)
                record["body_statement_markers"] = sorted(markers)
                record["branch_measurement"] = bool(coverage.get("meta", {}).get("branch_coverage", False)) and file_coverage is not None
                record["branches_executed"] = [arc for arc in (file_coverage or {}).get("executed_branches", []) if arc[0] in known_body]
                record["branches_missing"] = [arc for arc in (file_coverage or {}).get("missing_branches", []) if arc[0] in known_body]
                rows.append(record)
    gap_rows = [{"file": r["file"], "function": r["function"], "line": r["line"]} for r in rows if r["source_review"] != "reviewed"]
    unobserved = [{"file": r["file"], "function": r["function"], "line": r["line"], "evidence": r["execution_evidence"]} for r in rows if r["execution_evidence"] != "body_execution_observed"]
    summary = {
        "schema_version": 1,
        "scope": "All AST function/async-function definitions, methods, nested definitions and lambdas in mlguess/ (excluding tests/) and applications/; examples and notebooks are outside this inventory.",
        "method": [
            "Source-review evidence is explicit ledger reviewed:true, an explicit source-review status/note, or an exact-expression supplemental lambda review. Test presence alone does not imply source review.",
            "Execution evidence intersects coverage.py executed lines with the function's own body; definition lines, decorators, docstrings, nested definitions/classes and lambda-expression lines are excluded.",
            "A covered definition at import time never establishes invocation. One-line functions sharing their definition line may remain indeterminate.",
            "Lambdas are inventoried but their invocation is indeterminate from line coverage because creation and invocation may share a line.",
            "Observed body execution can come from a mocked or error-path test; see reviewer notes. It does not prove correctness, all branches, native platforms, GPU availability, or distributed runtime execution.",
            "Branch arcs are assigned only when their source belongs to the function's own covered body lines; absent branch measurement is never reported as complete coverage.",
        ],
        "coverage_source": {"file": coverage_file, "sha256": hashlib.sha256(coverage_path.read_bytes()).hexdigest(), "metadata": coverage.get("meta", {})},
        "review_ledger_sources": ledger_metadata,
        "missing_review_ledgers": missing_ledgers,
        "source_files": source_files,
        "counts": {"files": len(source_files), "functions_and_lambdas": len(rows), "kind": dict(Counter(r["kind"] for r in rows)), "source_review": dict(Counter(r["source_review"] for r in rows)), "execution_evidence": dict(Counter(r["execution_evidence"] for r in rows))},
        "source_review_gaps": gap_rows,
        "execution_not_established": unobserved,
        "functions": rows,
    }
    return rows, summary


def self_check():
    """Small regression checks for the audit's central false-positive risk."""
    source = 'def outer():\n    "doc"\n    def inner():\n        return 1\n    return inner\n\ndef one_line(): return 2\n'
    visitor = Inventory("synthetic.py")
    visitor.visit(ast.parse(source))
    records = {record["function"]: record for record in visitor.records}
    assert records["outer"]["_body_lines"] == {5}
    assert records["outer.inner"]["_body_lines"] == {4}
    assert not records["one_line"]["_body_lines"]
    # Import-only coverage (def lines 1 and 7) cannot establish either invocation.
    assert not ({1, 7} & records["outer"]["_body_lines"])
    assert not ({4} & records["outer"]["_body_lines"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--coverage", default="validation/audit-coverage.json")
    parser.add_argument("--ledger", action="append", help="Repeat to override the default review ledger list")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("Function-audit self-check passed.")
        return
    root = args.root.resolve()
    rows, summary = build(root, args.coverage, args.ledger or DEFAULT_LEDGERS)
    destination = root / "validation"
    destination.mkdir(exist_ok=True)
    (destination / "function-audit-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (destination / "function-audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["file", "function"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    print(json.dumps(summary["counts"], indent=2))
    if summary["source_review_gaps"]:
        print(f"Explicit source-review gaps: {len(summary['source_review_gaps'])}; see function-audit-summary.json.")


if __name__ == "__main__":
    main()
