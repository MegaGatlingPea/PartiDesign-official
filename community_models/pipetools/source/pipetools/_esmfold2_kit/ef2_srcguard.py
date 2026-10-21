"""Required upstream API checks for the adapted inference kernels.

Exact upstream source validation lives in scripts/validate_esmfold2_upstream_sources.py.
Production checks callable availability; numerical compatibility is accepted
offline for the installed software version.
"""
import importlib

LEVER_SOURCES = {                                                         # lever -> the upstream functions its code re-issues or wraps with assumptions on their statements
    "ls": (("transformers.models.esmfold2.modeling_esmfold2", "ESMFold2Model._run_one_loop"),),
    "rg": (("transformers.models.esmfold2.modeling_esmfold2", "ESMFold2Model._run_one_loop"),),
    "m15": (("transformers.models.esmfold2.modeling_esmfold2", "MSAEncoderBlock.forward"), ("transformers.models.esmfold2.modeling_esmfold2", "PairTransition.forward")),
    "m16": (("transformers.models.esmfold2.modeling_esmfold2", "MSAEncoderBlock.forward"), ("transformers.models.esmfold2.modeling_esmfold2_common", "MSAPairWeightedAveraging.forward")),
    "m17": (("transformers.models.esmfold2.modeling_esmfold2", "MSAEncoderBlock.forward"), ("transformers.models.esmfold2.modeling_esmfold2_common", "OuterProductMean.forward")),
    "mh": (("transformers.models.esmfold2.modeling_esmfold2_common", "maybe_subsample_msa"), ("transformers.models.esmfold2.modeling_esmfold2", "MSAEncoder.forward"), ("transformers.models.esmfold2.modeling_esmfold2", "ESMFold2Model._run_one_loop")),
    "t15": (("transformers.models.esmfold2.modeling_esmfold2_common", "Transition.forward"),),
    "t15msa": (("transformers.models.esmfold2.modeling_esmfold2", "PairTransition.forward"), ("transformers.models.esmfold2.modeling_esmfold2", "MSAEncoderBlock.forward")),
    "t16": (("transformers.models.esmfold2.modeling_esmfold2_common", "Transition.forward"), ("transformers.models.esmfold2.modeling_esmfold2", "PairTransition.forward")),
    "t6s": (("transformers.models.esmfold2.modeling_esmfold2_common", "Transition.forward"),),
    "t6i": (("transformers.models.esmfold2.modeling_esmfold2_common", "Transition.forward"),),
    "xtr": (("transformers.models.esmfold2.modeling_esmfold2", "PairTransition.forward"),),
    "trimul": (("transformers.models.esmfold2.modeling_esmfold2_common", "TriangleMultiplicativeBlock.forward"),),
    "glue": (("transformers.models.esmfold2.modeling_esmfold2_common", "TriangleMultiplicativeBlock.forward"),),
    "ax": (("transformers.models.esmfold2.modeling_esmfold2_common", "SWAAtomBlock.forward"), ("transformers.models.esmfold2.modeling_esmfold2_common", "SWA3DRoPEAttention.forward"), ("transformers.models.esmfold2.modeling_esmfold2_common", "ESMFold2AtomEncoder.forward"), ("transformers.models.esmfold2.modeling_esmfold2_common", "ESMFold2AtomDecoder.forward")),
    "af": (("transformers.models.esmfold2.modeling_esmfold2_common", "SWAAtomBlock.forward"), ("transformers.models.esmfold2.modeling_esmfold2_common", "SWA3DRoPEAttention.forward"), ("transformers.models.esmfold2.modeling_esmfold2_common", "ESMFold2AtomEncoder.forward"), ("transformers.models.esmfold2.modeling_esmfold2_common", "ESMFold2AtomDecoder.forward")),
    "ro": (("transformers.models.esmfold2.modeling_esmfold2_common", "DiffusionStructureHead.sample"), ("transformers.models.esmfold2.modeling_esmfold2_common", "DiffusionStructureHead._center_random_augmentation"), ("transformers.models.esmfold2.modeling_esmfold2_common", "DiffusionStructureHead._weighted_rigid_align"), ("transformers.models.esmfold2.modeling_esmfold2_common", "DiffusionModule.forward")),
    "kd": (("transformers.models.esmfold2.modeling_esmfold2_common", "DiffusionStructureHead._weighted_rigid_align"),),
    "dit": (("transformers.models.esmfold2.modeling_esmfold2_common", "DiffusionTransformer.forward"), ("transformers.models.esmfold2.modeling_esmfold2_common", "AttentionPairBias.forward"), ("transformers.models.esmfold2.modeling_esmfold2_common", "ConditionedTransitionBlock.forward"), ("transformers.models.esmfold2.modeling_esmfold2_common", "DiffusionConditioning.forward"), ("transformers.models.esmfold2.modeling_esmfold2_common", "DiffusionModule.forward")),
    "fz": (("esm.models.esmfold2.paired_msa", "msa_to_res_type_and_deletions"), ("esm.models.esmfold2.paired_msa", "construct_paired_msa")),
}

class StockMismatch(RuntimeError):
    """A required upstream API is missing or is no longer callable."""


def check(lever, keys=None):
    keys = list(keys if keys is not None else LEVER_SOURCES.get(lever, ()))
    for module_name, qualname in keys:
        value = importlib.import_module(module_name)
        try:
            for part in qualname.split("."):
                value = getattr(value, part)
        except AttributeError as error:
            raise StockMismatch(f"{lever}: required API {module_name}:{qualname} is unavailable") from error
        if not callable(value):
            raise StockMismatch(f"{lever}: required API {module_name}:{qualname} is not callable")
    return keys


def check_many(levers):
    items = [k for k, enabled in levers.items() if enabled] if isinstance(levers, dict) else list(levers)
    return {lever: check(lever) for lever in items if lever in LEVER_SOURCES}
