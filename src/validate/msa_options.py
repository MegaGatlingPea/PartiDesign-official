"""Shared MSA command overrides; no model or search imports."""
from __future__ import annotations


def cli_settings(args):
    fields = {
        'mode': 'msa', 'source': 'msa_source', 'server_url': 'msa_server_url',
        'cache_dir': 'msa_cache_dir', 'cache_profile': 'msa_profile',
        'database_dir': 'msa_database_dir', 'binaries_dir': 'msa_binaries_dir',
        'search_workers': 'msa_workers', 'cpu_budget': 'msa_cpus',
        'jackhmmer_n_cpu': 'jackhmmer_cpus', 'nhmmer_n_cpu': 'nhmmer_cpus',
        'timeout_seconds': 'msa_timeout', 'templates': 'msa_templates',
        'max_template_date': 'msa_template_date',
    }
    result = {field: getattr(args, attr) for field, attr in fields.items()
              if getattr(args, attr, None) is not None}
    if isinstance(result.get('templates'), str):
        result['templates'] = result['templates'] == 'true'
    if getattr(args, 'msa_refresh', False):
        result['refresh'] = True
    return result


def merge_shared(base, overlay):
    """Lower optional shared settings while preserving explicit model overrides."""
    from .af3_msa import validate_msa_settings
    from .config import merge
    shared = overlay.get('msa', {})
    validate_msa_settings(shared)
    result = merge(base, {k: v for k, v in overlay.items() if k != 'msa'})
    for group in ('esmfold2', 'af3', 'mol1', 'mol1_constraint', 'boltz2'):
        if group not in result:
            continue
        explicit = overlay.get(group, {}).get('msa', {})
        result[group]['msa'] = merge(result[group]['msa'], {**shared, **explicit})
    return result


def normalize_esmfold2_msa(settings):
    """Reuse the shared MSA schema for the protein-only ESMFold2 consumer."""
    from .af3_msa import validate_msa_settings

    result = validate_msa_settings(settings)
    result['templates'] = False
    return result


def query_identity(settings, modality, sequence):
    """Search semantics, excluding scheduling and cache access policy."""
    return (settings['cache_dir'], settings['source'],
            settings.get('server_url') if settings['source'] == 'colabfold' else None,
            settings.get('database_revision'), settings.get('cache_profile'),
            settings.get('templates') if modality == 'protein' else False,
            settings.get('max_template_date') if modality == 'protein' else None,
            modality, sequence)
