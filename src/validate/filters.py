"""Named, source-attributed criteria over complete, same-prediction metrics."""
from __future__ import annotations

import math
import operator
from pathlib import Path

from .config import METHODS, load_yaml
from .storage import nested_record_valid, record_complete

OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le,
       "==": operator.eq}


def load_presets(path: Path) -> dict:
    presets = load_yaml(path).get("presets")
    if not isinstance(presets, dict) or not presets:
        raise ValueError("Preset file must have a nonempty presets mapping")
    for name, preset in presets.items():
        if not isinstance(name, str) or not isinstance(preset, dict):
            raise ValueError("Every named preset must be a mapping")
        if not isinstance(preset.get("methods"), list) or not preset["methods"] or any(m not in METHODS for m in preset["methods"]):
            raise ValueError(f"Invalid methods for {name}")
        if preset.get("reduction") not in {"all_and_mean", "any_prediction", "best_ranked", "any_af2_pair"}:
            raise ValueError(f"Invalid reduction for {name}")
        if not isinstance(preset.get("rules"), list) or not preset["rules"]:
            raise ValueError(f"Empty rules for {name}")
        for rule in preset["rules"]:
            if not isinstance(rule, dict):
                raise ValueError(f"Every rule in {name} must be a mapping")
            if rule.get("op") not in OPS or not isinstance(rule.get("metric"), str):
                raise ValueError(f"Invalid rule in {name}")
            if rule.get("scope", "complex") not in {"complex", "monomer", "paired", "prediction"}:
                raise ValueError(f"Invalid metric scope in {name}")
            if not isinstance(rule.get("value"), (int, float)) or not math.isfinite(rule["value"]):
                raise ValueError(f"Invalid threshold in {name}")
    return presets


def _number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _test(metrics: dict, rules: list[dict]) -> tuple[str, list[str]]:
    missing, failed = [], []
    for rule in rules:
        key = rule["metric"]
        value = metrics.get(key)
        if not _number(value):
            missing.append(key)
        elif not OPS[rule["op"]](value, rule["value"]):
            failed.append(f"{key}={value} does not satisfy {rule['op']} {rule['value']}")
    if missing:
        return "unavailable", [f"missing required metric: {x}" for x in missing] + failed
    return ("failed", failed) if failed else ("passed", [])


def evaluate(record: dict, name: str, preset: dict) -> dict:
    method = record.get("method")
    result = {"criteria": name, "status": "unavailable", "reasons": []}
    if method not in preset["methods"]:
        return {**result, "status": "not_applicable", "reasons": [f"preset does not apply to {method}"]}
    if record.get("status") == "not_applicable":
        return {**result, "status": "not_applicable", "reasons": [record.get("error", "method capability mismatch")]}
    if record.get("status") != "complete":
        return {**result, "reasons": ["requested sequence validation stages are not complete"]}
    if not nested_record_valid(record):
        return {**result, "reasons": ["malformed nested prediction/metric/score state"]}
    expected = record.get("expected_predictions", [])
    predictions = record.get("predictions", {})
    if not expected or any(predictions.get(k, {}).get("status") != "complete" for k in expected):
        return {**result, "reasons": ["required predictions incomplete or failed"]}
    if any(predictions[k].get("protocol") not in ("complex", "monomer") for k in expected):
        return {**result, "reasons": ["prediction protocol must be complex or monomer"]}
    if any(predictions[k].get("metrics_status") != "complete" for k in expected):
        return {**result, "reasons": ["prediction metrics are not complete"]}
    if method == "af2_ig":
        pairs = {key.removeprefix("complex_") for key in expected if key.startswith("complex_")}
        if set(record.get("paired_metrics", {})) != pairs:
            return {**result, "reasons": ["not every configured AF2 model/sample has a paired comparison"]}
    by_scope = {"complex": [], "monomer": [], "paired": [], "prediction": []}
    for key in expected:
        prediction = predictions[key]
        by_scope.setdefault(prediction["protocol"], []).append((key, prediction.get("metrics", {})))
        by_scope["prediction"].append((key, prediction.get("metrics", {})))
    for key, metrics in record.get("paired_metrics", {}).items():
        by_scope["paired"].append((key, metrics))
    rules_by_scope = {}
    for rule in preset["rules"]:
        rules_by_scope.setdefault(rule.get("scope", "complex"), []).append(rule)
    statuses, reasons, passed = [], [], []
    reduction = preset["reduction"]
    if reduction == "any_af2_pair":
        if method != "af2_ig" or not set(rules_by_scope).issubset({"complex", "monomer", "paired"}):
            raise ValueError("any_af2_pair requires AF2 complex/monomer/paired rules")
        for key, complex_metrics in by_scope["complex"]:
            suffix = key.removeprefix("complex_")
            group = {
                "complex": complex_metrics,
                "monomer": predictions.get("monomer_" + suffix, {}).get("metrics", {}),
                "paired": record.get("paired_metrics", {}).get(suffix, {}),
            }
            pair_status, pair_reasons = [], []
            for scope, rules in rules_by_scope.items():
                status, why = _test(group[scope], rules)
                pair_status.append(status)
                pair_reasons.extend(f"{key} {scope}: {item}" for item in why)
            statuses.extend(pair_status)
            reasons.extend(pair_reasons)
            if all(status == "passed" for status in pair_status):
                passed.append(key)
        result["status"] = "passed" if passed else "unavailable" if "unavailable" in statuses else "failed"
    elif reduction == "all_and_mean":
        for scope, rules in rules_by_scope.items():
            group = by_scope[scope]
            if not group:
                statuses.append("unavailable")
                reasons.append(f"missing {scope} predictions/paired comparisons")
                continue
            # Pair completeness is checked independently; never average around a failed model.
            expected_pairs = {key.removeprefix("complex_") for key, _ in by_scope["complex"]}
            if scope == "paired" and {key for key, _ in group} != expected_pairs:
                statuses.append("unavailable")
                reasons.append("not every complex has its paired monomer comparison")
            for key, metrics in group:
                status, why = _test(metrics, rules)
                statuses.append(status)
                reasons.extend(f"{key}: {item}" for item in why)
            means = {}
            for rule in rules:
                values = [metrics.get(rule["metric"]) for _, metrics in group]
                means[rule["metric"]] = sum(values) / len(values) if all(_number(v) for v in values) else None
            status, why = _test(means, rules)
            statuses.append(status)
            reasons.extend(f"mean {scope}: {item}" for item in why)
        result["status"] = "unavailable" if "unavailable" in statuses else "failed" if "failed" in statuses else "passed"
        if result["status"] == "passed":
            passed = expected
    else:
        # A conjunction is evaluated on a single prediction; never mix best metrics across samples.
        if len(rules_by_scope) != 1 or not set(rules_by_scope).issubset({"complex", "monomer", "prediction"}):
            raise ValueError("Prediction selectors require one complex, monomer, or prediction scope")
        scope = next(iter(rules_by_scope))
        candidates = by_scope[scope]
        if not candidates:
            return {**result, "reasons": [f"missing {scope} predictions"]}
        if reduction == "best_ranked":
            if any(not _number(m.get("iptm")) or not _number(m.get("ptm")) for _, m in candidates):
                return {**result, "reasons": ["ranking requires native iptm and ptm for every prediction"]}
            candidates = sorted(candidates, key=lambda pair: (-0.8 * pair[1]["iptm"] - 0.2 * pair[1]["ptm"], pair[0]))[:1]
        for key, metrics in candidates:
            status, why = _test(metrics, rules_by_scope[scope])
            statuses.append(status)
            reasons.extend(f"{key}: {item}" for item in why)
            if status == "passed":
                passed.append(key)
        result["status"] = "passed" if passed else "unavailable" if "unavailable" in statuses else "failed"
    result.update(reasons=reasons, passing_prediction_keys=passed)
    return result


def _unfiltered_completion(record: dict) -> dict:
    """An execution-only cohort; no structural quality threshold is applied."""
    result = {"criteria": "unfiltered", "status": "unavailable", "reasons": [],
              "passing_prediction_keys": []}
    if record.get("status") == "not_applicable":
        return {**result, "status": "not_applicable", "reasons": ["method capability mismatch"]}
    if record.get("status") != "complete":
        return {**result, "reasons": ["requested validation stages are not complete"]}
    expected = record.get("expected_predictions")
    if (record.get("method") not in METHODS or not isinstance(expected, list) or not expected
            or any(not isinstance(key, str) or not key for key in expected)
            or len(expected) != len(set(expected)) or not nested_record_valid(record)):
        return {**result, "reasons": ["malformed validation prediction state"]}
    if not record_complete(record, expected, method=record["method"]):
        return {**result, "reasons": ["required predictions, metrics or artifacts are incomplete"]}
    return {**result, "status": "passed", "passing_prediction_keys": list(expected)}


def apply_presets(record: dict, names: list[str], presets: dict, *, unfiltered=False) -> dict:
    unknown = set(names) - presets.keys()
    if unknown:
        raise ValueError(f"Unknown criteria: {sorted(unknown)}")
    record["decisions"] = {name: evaluate(record, name, presets[name]) for name in names}
    if unfiltered and not names:
        record["decisions"]["unfiltered"] = _unfiltered_completion(record)
    return record
