"""Reward function for Mind2Web step prediction.

Plugged into the lasgroup/SDPO reward registry by `data/install_reward.sh`.
File is copied into upstream/verl/utils/reward_score/feedback/mind2web.py at
prepare time, and the dispatcher in __init__.py is patched to call us when
`data_source == "mind2web"`.

Action format (both predicted and gold):
    [<element_type>] <element_text> -> <ACTION>: <value-if-any>

Examples:
    [button] Sign in -> CLICK
    [searchbox] Find a location -> TYPE: Boston
    [combobox] Time -> SELECT: 5:00 PM

Reward decomposition (mirrors Mind2Web's official metrics):
    + 0.2 if the predicted line matches the format
    + 0.2 if the predicted ACTION (CLICK/TYPE/SELECT/etc.) matches gold
    + 0.3 if the element_text matches gold (substring-tolerant)
    + 0.3 if the value matches gold (when applicable; full credit if no value
                                     expected and none predicted)
    -> sum in [0, 1]

`acc` is 1.0 only when ACTION + element + value all match exactly (Mind2Web's
"step success" metric).
"""

from __future__ import annotations

import re
from typing import Optional


_ACTION_LINE = re.compile(
    r"^\s*\[(?P<etype>[^\]]+)\]\s*(?P<etext>.*?)\s*->\s*(?P<action>[A-Z]+)\s*(?::\s*(?P<value>.*))?\s*$"
)


def _parse_action_line(text: str) -> Optional[dict]:
    """Parse the FIRST well-formed action line in `text`. Returns None if none."""
    if not text:
        return None
    for raw in text.strip().splitlines():
        m = _ACTION_LINE.match(raw)
        if m:
            return {
                "etype": (m.group("etype") or "").strip().lower(),
                "etext": (m.group("etext") or "").strip().lower(),
                "action": (m.group("action") or "").strip().upper(),
                "value": (m.group("value") or "").strip().lower(),
            }
    return None


def _element_match(pred: str, gold: str) -> bool:
    """Tolerant element-text match: case-insensitive, substring either way."""
    if not pred or not gold:
        return pred == gold
    return pred == gold or pred in gold or gold in pred


def compute_score(solution: str, ground_truth: str, extra_info=None) -> dict:
    gold = _parse_action_line(ground_truth)
    pred = _parse_action_line(solution)

    if gold is None:
        # Malformed gold should never happen — bail safely.
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": solution[:200] if solution else "",
            "incorrect_format": 1,
            "feedback": "malformed ground_truth",
        }

    if pred is None:
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": solution[:200] if solution else "",
            "incorrect_format": 1,
            "feedback": (
                "Output did not contain a valid action line. Expected format:\n"
                "[<element_type>] <element_text> -> <ACTION>: <value>"
            ),
        }

    score = 0.2  # well-formatted
    feedback_parts = ["format ok"]

    action_ok = pred["action"] == gold["action"]
    if action_ok:
        score += 0.2
        feedback_parts.append(f"action ok ({gold['action']})")
    else:
        feedback_parts.append(
            f"action mismatch: predicted {pred['action']!r}, expected {gold['action']!r}"
        )

    element_ok = _element_match(pred["etext"], gold["etext"]) and (
        not gold["etype"] or pred["etype"] == gold["etype"]
    )
    if element_ok:
        score += 0.3
        feedback_parts.append(f"element ok ([{gold['etype']}] {gold['etext']})")
    else:
        feedback_parts.append(
            f"element mismatch: predicted [{pred['etype']}] {pred['etext']!r}, "
            f"expected [{gold['etype']}] {gold['etext']!r}"
        )

    if not gold["value"]:
        # No value expected — give full credit if model also didn't emit one.
        if not pred["value"]:
            score += 0.3
            feedback_parts.append("no value expected, none given")
        else:
            feedback_parts.append(f"unexpected value: {pred['value']!r}")
    else:
        if pred["value"] == gold["value"]:
            score += 0.3
            feedback_parts.append(f"value ok ({gold['value']!r})")
        else:
            feedback_parts.append(
                f"value mismatch: predicted {pred['value']!r}, expected {gold['value']!r}"
            )

    acc = 1.0 if (action_ok and element_ok and (
        gold["value"] == pred["value"] if gold["value"] else not pred["value"]
    )) else 0.0

    return {
        "score": float(score),
        "acc": float(acc),
        "pred": solution[:200] if solution else "",
        "incorrect_format": 0,
        "feedback": "; ".join(feedback_parts),
    }


# ---------------------------------------------------------------------------
# Self-test (run with: python mind2web_reward.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cases = [
        # (gold, pred, expected_acc, label)
        (
            "[button] Sign in -> CLICK",
            "[button] Sign in -> CLICK",
            1.0, "exact match",
        ),
        (
            "[searchbox] Find a location -> TYPE: Boston",
            "[searchbox] Find a location -> TYPE: boston",
            1.0, "case-insensitive value",
        ),
        (
            "[combobox] Time -> SELECT: 5:00 PM",
            "[combobox] Time -> CLICK",
            0.0, "wrong action",
        ),
        (
            "[button] Sign in -> CLICK",
            "Thought: I should click sign in.\n[button] Sign in -> CLICK",
            1.0, "extra reasoning lines",
        ),
        (
            "[button] Sign in -> CLICK",
            "I have no idea",
            0.0, "no action line",
        ),
    ]

    print("self-test:")
    for gold, pred, expected_acc, label in cases:
        r = compute_score(pred, gold)
        ok = "OK " if r["acc"] == expected_acc else "FAIL"
        print(f"  [{ok}] {label}: score={r['score']:.2f} acc={r['acc']:.1f}")
