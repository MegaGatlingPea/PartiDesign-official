"""Checkpoint preflight for the antibody template feature contract."""
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..core.records import DesignConfigError


def validate_antibody_checkpoint_metadata(document: Mapping[str, Any]) -> dict[str, int]:
    """Validate metadata and embedding shapes without constructing an FM model."""
    try:
        import torch
        from ..model.checkpoint import _checkpoint_experiment_config
        from ..model.networks.feature import FeatureFactory

        network = _checkpoint_experiment_config(document)['network']
        template = network.get('soft_template')
        if not isinstance(template, Mapping) or template.get('enabled') is not True:
            raise ValueError('network.soft_template.enabled must be true')
        contracts = [
            ('feats_seq', 'soft_template_single', 'seq', 'token_dim', 11,
             'nn.init_repr_factory.linear_out.weight'),
            ('feats_pair_repr', 'soft_template_pair', 'pair', 'pair_repr_dim', 111,
             'nn.pair_repr_builder.init_repr_factory.linear_out.weight'),
        ]
        state = document['state_dict']
        if not isinstance(state, Mapping):
            raise TypeError('checkpoint.state_dict must be a mapping')
        result = {}
        for field, feature, mode, output_dim, tail_dim, key in contracts:
            names = network.get(field)
            if not isinstance(names, (list, tuple)) or names.count(feature) != 1 or names[-1] != feature:
                raise ValueError(f'{field} must end with exactly one {feature}')
            with torch.device('meta'):
                factory = FeatureFactory(feats=names, dim_feats_out=int(network[output_dim]),
                    use_ln_out=False, mode=mode, feature_registry='flow', **network)
            if factory.template_tail_dim != tail_dim:
                raise ValueError(f'{feature} must have {tail_dim} channels')
            weight = state.get(key)
            expected = (int(network[output_dim]), factory.raw_dim)
            if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != expected:
                raise ValueError(f'{key} must have shape {expected}')
            result[mode + '_raw_dim'] = factory.raw_dim
            result[mode + '_template_dim'] = tail_dim
        return result
    except (KeyError, TypeError, ValueError, AssertionError, RuntimeError) as error:
        raise DesignConfigError(f'antibody checkpoint template ABI preflight failed: {error}') from error


def validate_antibody_checkpoint(
    path: str | Path, *, task_formats: Sequence[str] = ('antibody',),
    variant: str | None = "antibody", require_bundle: bool = False,
) -> dict[str, int] | None:
    # Normal requests retain their existing checkpoint path, including legacy models.
    if 'antibody' not in task_formats:
        return None
    try:
        from ..model.checkpoint import _read_selected_checkpoint
        document, _source, _member = _read_selected_checkpoint(
            path, variant=variant, require_bundle=require_bundle,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise DesignConfigError(f'antibody checkpoint template ABI preflight failed: {error}') from error
    return validate_antibody_checkpoint_metadata(document)
