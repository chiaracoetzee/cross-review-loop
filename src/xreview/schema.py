"""Strict JSON Schema and validation for reviewer replies."""

from __future__ import annotations

from typing import Any

SEVERITIES = ("blocker", "major", "minor", "nit")
SEVERITY_RANK = {"blocker": 4, "major": 3, "minor": 2, "nit": 1}
CONFIDENCES = ("high", "medium", "low")
VERDICTS = ("concede", "maintain", "fix_ok", "fix_incomplete", "answer", "second")

# Written to OpenAI's strict structured-output rules (every field required,
# additionalProperties: false), which is the strictest of the three harnesses.
REVIEWER_REPLY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["findings", "responses", "summary"],
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "key",
                    "title",
                    "severity",
                    "confidence",
                    "file",
                    "line",
                    "evidence",
                    "failure_scenario",
                    "suggested_fix",
                ],
                "properties": {
                    "key": {"type": "string"},
                    "title": {"type": "string"},
                    "severity": {"type": "string", "enum": list(SEVERITIES)},
                    "confidence": {"type": "string", "enum": list(CONFIDENCES)},
                    "file": {"type": "string"},
                    "line": {"type": "integer"},
                    "evidence": {"type": "string"},
                    "failure_scenario": {"type": "string"},
                    "suggested_fix": {"type": "string"},
                },
            },
        },
        "responses": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ledger_id", "verdict", "argument"],
                "properties": {
                    "ledger_id": {"type": "string"},
                    "verdict": {"type": "string", "enum": list(VERDICTS)},
                    "argument": {"type": "string"},
                },
            },
        },
        "summary": {"type": "string"},
    },
}


def validate_reply_payload(payload: Any) -> list[str]:
    """Return a list of schema validation errors (empty if valid)."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["reply must be a JSON object"]

    expected_top = {"findings", "responses", "summary"}
    missing_top = expected_top - set(payload.keys())
    extra_top = set(payload.keys()) - expected_top
    if missing_top:
        errors.append(f"missing top-level fields: {sorted(missing_top)}")
    if extra_top:
        errors.append(f"unexpected top-level fields: {sorted(extra_top)}")
    if errors:
        return errors

    if not isinstance(payload["summary"], str):
        errors.append("summary must be a string")

    findings = payload["findings"]
    if not isinstance(findings, list):
        errors.append("findings must be an array")
    else:
        req_f = {
            "key",
            "title",
            "severity",
            "confidence",
            "file",
            "line",
            "evidence",
            "failure_scenario",
            "suggested_fix",
        }
        for i, item in enumerate(findings):
            if not isinstance(item, dict):
                errors.append(f"findings[{i}] must be an object")
                continue
            miss = req_f - set(item.keys())
            extra = set(item.keys()) - req_f
            if miss:
                errors.append(f"findings[{i}] missing fields: {sorted(miss)}")
            if extra:
                errors.append(f"findings[{i}] extra fields: {sorted(extra)}")
            if miss or extra:
                continue
            for str_field in (
                "key",
                "title",
                "file",
                "evidence",
                "failure_scenario",
                "suggested_fix",
            ):
                if not isinstance(item[str_field], str):
                    errors.append(f"findings[{i}].{str_field} must be a string")
            if item["severity"] not in SEVERITIES:
                errors.append(f"findings[{i}].severity invalid: {item['severity']!r}")
            if item["confidence"] not in CONFIDENCES:
                errors.append(f"findings[{i}].confidence invalid: {item['confidence']!r}")
            if isinstance(item["line"], bool) or not isinstance(item["line"], int):
                errors.append(f"findings[{i}].line must be an integer")

    responses = payload["responses"]
    if not isinstance(responses, list):
        errors.append("responses must be an array")
    else:
        req_r = {"ledger_id", "verdict", "argument"}
        for i, item in enumerate(responses):
            if not isinstance(item, dict):
                errors.append(f"responses[{i}] must be an object")
                continue
            miss = req_r - set(item.keys())
            extra = set(item.keys()) - req_r
            if miss:
                errors.append(f"responses[{i}] missing fields: {sorted(miss)}")
            if extra:
                errors.append(f"responses[{i}] extra fields: {sorted(extra)}")
            if miss or extra:
                continue
            if not isinstance(item["ledger_id"], str):
                errors.append(f"responses[{i}].ledger_id must be a string")
            if item["verdict"] not in VERDICTS:
                errors.append(f"responses[{i}].verdict invalid: {item['verdict']!r}")
            if not isinstance(item["argument"], str):
                errors.append(f"responses[{i}].argument must be a string")

    return errors


def severity_max(severities: list[str]) -> str:
    """Return the highest severity among the given valid severity strings."""
    if not severities:
        return "nit"
    return max(severities, key=lambda s: SEVERITY_RANK.get(s, 0))
