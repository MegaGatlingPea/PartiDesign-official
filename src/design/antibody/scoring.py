"""Antibody interface aggregation without changing the structure scorer."""
from dataclasses import replace
import math


def antibody_binder_chains(candidate):
    entries = getattr(candidate, "antibody_metadata", {}).get("entries", ())
    chains = tuple(dict.fromkeys(str(e["chain_id"]) for e in entries
                                if e.get("role") in {"VH", "VL", "VHH"}))
    if not chains:
        raise ValueError("antibody scoring requires compiled chain roles")
    return chains


def _aggregate(result, interfaces, scope):
    if result.status == "error":
        return replace(result, score=None, mean_log_density=None, density_sum=0., pair_count=0, scope=scope)
    if not interfaces:
        return replace(result, score=None, mean_log_density=None, density_sum=0.,
                       pair_count=0, status="no_interface", error=None,
                       interfaces=[], scope=scope)
    if any(i.status == "error" for i in interfaces):
        return replace(result, score=None, mean_log_density=None, density_sum=0.,
                       pair_count=0, status="error",
                       error="; ".join(i.error or "interface scoring failed" for i in interfaces if i.status == "error"),
                       interfaces=list(interfaces), scope=scope)
    count = sum(i.pair_count for i in interfaces)
    if not count:
        return replace(result, score=None, mean_log_density=None, density_sum=0.,
                       pair_count=0, status="no_contact", error=None,
                       interfaces=list(interfaces), scope=scope)
    mean = sum(i.mean_log_density * i.pair_count for i in interfaces if i.pair_count) / count
    if not math.isfinite(mean):
        raise RuntimeError("nonfinite antibody interface density")
    return replace(result, score=mean + math.log(count), mean_log_density=mean,
                   density_sum=sum(i.density_sum for i in interfaces), pair_count=count,
                   status="ok", error=None, interfaces=list(interfaces), scope=scope)


def antibody_score_components(result, binder_chains):
    """Aggregate contact pairs, never equally average per-interface scores."""
    binder = set(binder_chains)
    antigen, internal = [], []
    for interface in result.interfaces:
        left = interface.partner_a in binder
        right = interface.partner_b in binder
        if left != right:
            antigen.append(interface)
        elif left and right:
            internal.append(interface)
    return (_aggregate(result, antigen, "antibody_antigen"),
            _aggregate(result, internal, "antibody_internal"))


def antibody_score_record(result, binder_chains):
    antigen, internal = antibody_score_components(result, binder_chains)
    if len(set(binder_chains)) < 2:
        internal = replace(internal, score=None, mean_log_density=None, density_sum=0.,
                           pair_count=0, status="no_interface", error=None, interfaces=[])
    values = {}
    for name, component in (("antigen", antigen), ("internal", internal)):
        prefix = f"partiscore_{name}"
        values[prefix] = component.score
        values[f"{prefix}_pair_count"] = (None if component.status in {"no_interface", "error"} else component.pair_count)
        values[f"{prefix}_status"] = component.status
        values[f"{prefix}_error"] = component.error
    return values
