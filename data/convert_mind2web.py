"""
Convert osunlp/Mind2Web into the format expected by lasgroup/SDPO's data
pipeline (`upstream/data/preprocess.py`). Emits one (prompt, gold-action)
example per step inside each task.

Output schema matches the union of fields produced by upstream/data/load_dataset.py:
    idx, kind, dataset, answer, prompt, description, tests, system, embedding, elo

Reward shape: exact match between model's emitted next-action string and `answer`
(i.e. the gold `action_reprs[step]`). Plug in via the upstream verifier hooks
when invoking `data/preprocess.py --data_source datasets/mind2web`.

Why per-step instead of full-trajectory:
- Keeps context length within Qwen3.5-9B's window (~32k)
- Cleaner credit assignment for GRPO/SDPO
- Matches how Mind2Web is conventionally trained (SEEACT-style step prediction)

Why no DOM:
- cleaned_html per step is 5–50k tokens. With history + 8 rollouts it blows context.
- We pass the task description + truncated action history as the "current state"
  proxy. The model must rely on the task semantics + prior actions to predict next.
- This is weaker supervision than full SEEACT but is enough for the comparison
  question (does SDPO converge faster than GRPO under matched conditions).
- Set --include-dom to fall back to truncated DOM if you want richer state.

Usage:
    pip install datasets pyarrow
    python data/convert_mind2web.py \
        --output-dir upstream/datasets/mind2web \
        --max-train-tasks 800 \
        --max-test-tasks 100
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterator


SYSTEM_PROMPT = (
    "You are a web agent. Given a high-level task and the actions you have "
    "already taken, output the single next action to take on the current page. "
    "Respond with exactly one line in this format:\n"
    "[<element_type>] <element_text> -> <ACTION>: <value-if-any>\n"
    "Examples:\n"
    "  [button] Sign in -> CLICK\n"
    "  [searchbox] Find a location -> TYPE: Boston\n"
    "  [combobox] Time -> SELECT: 5:00 PM"
)


def _format_action_history(prior_actions: list[str], max_chars: int = 3000) -> str:
    if not prior_actions:
        return "(none — this is the first action)"
    rendered = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(prior_actions))
    if len(rendered) > max_chars:
        rendered = rendered[-max_chars:]
        rendered = "...\n" + rendered
    return rendered


def _truncate_dom(html: str | None, budget_chars: int) -> str:
    if not html or budget_chars <= 0:
        return ""
    if len(html) <= budget_chars:
        return html
    head = html[: budget_chars // 2]
    tail = html[-budget_chars // 2 :]
    return f"{head}\n... [DOM truncated] ...\n{tail}"


def _build_prompt(
    task_description: str,
    website: str,
    prior_actions: list[str],
    dom_snippet: str = "",
) -> str:
    parts = [
        f"Website: {website}",
        f"Task: {task_description}",
        "",
        "Actions taken so far:",
        _format_action_history(prior_actions),
    ]
    if dom_snippet:
        parts += ["", "Current page (truncated DOM):", dom_snippet]
    parts += ["", "Next action:"]
    return "\n".join(parts)


def _iter_examples(
    task_records: Iterator[dict],
    dataset_name: str,
    include_dom: bool,
    dom_budget: int,
    start_idx: int = 0,
) -> Iterator[dict]:
    idx = start_idx
    for task in task_records:
        confirmed_task = (task.get("confirmed_task") or "").strip()
        website = task.get("website") or ""
        action_reprs = task.get("action_reprs") or []
        actions = task.get("actions") or []
        if not confirmed_task or not action_reprs:
            continue

        for step, gold_action in enumerate(action_reprs):
            prior = list(action_reprs[:step])
            dom_snippet = ""
            if include_dom and step < len(actions):
                raw_html = actions[step].get("cleaned_html")
                dom_snippet = _truncate_dom(raw_html, dom_budget)
            prompt = _build_prompt(confirmed_task, website, prior, dom_snippet)
            yield {
                "idx": idx,
                "kind": "mind2web",
                # short name — must match the dispatch key in
                # upstream/verl/utils/reward_score/feedback/__init__.py
                "dataset": "mind2web",
                "answer": gold_action.strip(),
                "prompt": prompt,
                "description": prompt,  # SDPO loader appends environment hint to description
                "tests": "-",
                "system": SYSTEM_PROMPT,
                "elo": "-",
                "embedding": [],
            }
            idx += 1


def write_jsonl(rows: Iterator[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            n += 1
    return n


def _load_via_hf(hf_id: str) -> Iterator[dict]:
    from datasets import load_dataset
    ds = load_dataset(hf_id, split="train")
    print(f"[load:hf] {len(ds)} tasks via {hf_id}")
    for row in ds:
        yield row


def _load_via_parquet(paths: list[Path]) -> Iterator[dict]:
    import pyarrow.parquet as pq
    total = 0
    for p in paths:
        t = pq.read_table(str(p))
        total += t.num_rows
        cols = t.column_names
        for i in range(t.num_rows):
            yield {c: t.column(c)[i].as_py() for c in cols}
    print(f"[load:parquet] {total} tasks across {len(paths)} files")


def _materialize(it: Iterator[dict], limit: int | None) -> list[dict]:
    out = []
    for row in it:
        out.append(row)
        if limit is not None and len(out) >= limit:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="upstream/datasets/mind2web", type=Path)
    ap.add_argument("--hf-id", default="osunlp/Mind2Web")
    ap.add_argument("--parquet", nargs="*", type=Path, default=None,
                    help="local parquet file(s) to read instead of HF (faster for testing)")
    ap.add_argument("--max-train-tasks", type=int, default=800,
                    help="cap on tasks (each yields ~10 step-examples)")
    ap.add_argument("--max-test-tasks", type=int, default=100)
    ap.add_argument("--include-dom", action="store_true",
                    help="include truncated cleaned_html as context (much longer prompts)")
    ap.add_argument("--dom-budget", type=int, default=3000,
                    help="max chars of DOM included per step when --include-dom")
    args = ap.parse_args()

    if args.parquet:
        source_iter = _load_via_parquet(args.parquet)
    else:
        print(f"[load] {args.hf_id}")
        source_iter = _load_via_hf(args.hf_id)

    n_total = args.max_test_tasks + args.max_train_tasks
    all_tasks = _materialize(source_iter, limit=n_total)
    print(f"[load] materialized {len(all_tasks)} tasks (asked for {n_total})")

    n_test = min(args.max_test_tasks, len(all_tasks))
    test_tasks = all_tasks[:n_test]
    train_tasks = all_tasks[n_test:]

    train_path = args.output_dir / "train.json"
    test_path = args.output_dir / "test.json"

    n_train_ex = write_jsonl(
        _iter_examples(iter(train_tasks), args.hf_id, args.include_dom,
                       args.dom_budget, start_idx=0),
        train_path,
    )
    n_test_ex = write_jsonl(
        _iter_examples(iter(test_tasks), args.hf_id, args.include_dom,
                       args.dom_budget, start_idx=10_000_000),
        test_path,
    )

    print(f"[write] {train_path}: {n_train_ex} step-examples from {len(train_tasks)} tasks")
    print(f"[write] {test_path}: {n_test_ex} step-examples from {len(test_tasks)} tasks")
    print()
    print("Next: cd upstream && python data/preprocess.py "
          f"--data_source datasets/mind2web")


if __name__ == "__main__":
    main()
