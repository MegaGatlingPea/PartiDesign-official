"""Prepare and pin shared context search evidence before GPU owners start."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .state import atomic_json, write_csv
from src.validate.methods import REFOLD_METHODS


def prepare_run(cfg, repo, work):
    from src.validate.msa_command import task_queries
    from src.validate.msa_options import query_identity
    from src.validate.af3_msa import AF3MSACache, resolve_msa_settings
    from src.validate.af3_msa_runtime import available_cpus, query_cost
    import json

    stages = [stage for task in cfg['tasks'].values()
              for stage in [*task['rounds'], *task['final'].get('validators', [task['final']])]]
    if not any(REFOLD_METHODS[stage['method']].msa and (
            REFOLD_METHODS[stage['method']].family != 'esmfold2'
            or stage['validation']['esmfold2'].get('msa', {}).get('mode', 'off') != 'off')
            for stage in stages):
        return [], []
    state = Path(work) / 'msa.json'
    if state.is_file():
        saved = json.loads(state.read_text())
        references = []
        for item in saved['references']:
            settings = dict(item['settings'], mode='cache-only', refresh=False)
            cache = AF3MSACache(repo, settings, readonly=True)
            ref = cache.reference(item['entry_id'], item['profile_id'], item['generation_id'])
            if ref is None:
                raise ValueError('Pinned run MSA/template cache is missing; restore its recorded generation')
            references.append(ref)
        return references, saved['rows']

    rows = task_queries(cfg)
    caches, requests, references, saved_refs = {}, {}, [], []
    for row in rows:
        settings = resolve_msa_settings(row.pop('settings'), repo)
        identity = tuple(sorted(settings.items()))
        if identity not in caches:
            caches[identity] = AF3MSACache(repo, settings)
        cache = caches[identity]
        row.update(cache_dir=str(cache.cache_dir), status='query-only')
        if not row['enabled']:
            continue
        key = query_identity(settings, row['modality'], row['sequence'])
        previous = requests.get(key)
        if previous is None or (settings['mode'] == 'auto' and previous[3]['mode'] != 'auto'):
            requests[key] = (cache, row['modality'], row['sequence'], settings)
        elif settings['refresh']:
            merged = dict(previous[3], refresh=True)
            requests[key] = (AF3MSACache(repo, merged), previous[1], previous[2], merged)
        row['_request'] = key

    # A template-on search also supplies template-off consumers' alignments.
    # Only an automatic template-on consumer authorizes that superset search.
    donors, aliases = {}, {}
    def alignment_key(settings, modality, sequence):
        return query_identity(dict(settings, templates=False, max_template_date=None), modality, sequence)
    for key, (_, modality, sequence, settings) in requests.items():
        if (modality == 'protein' and settings['templates'] and settings['mode'] == 'auto'
                and settings['cache_profile'] is None):
            donors.setdefault(alignment_key(settings, modality, sequence), key)
    for key, (_, modality, sequence, settings) in requests.items():
        if modality != 'protein' or settings['templates'] or settings['cache_profile'] is not None:
            continue
        donor = donors.get(alignment_key(settings, modality, sequence))
        if donor is None:
            continue
        aliases[key] = donor
        _, donor_modality, donor_sequence, donor_settings = requests[donor]
        if settings['refresh'] and not donor_settings['refresh']:
            merged = dict(donor_settings, refresh=True)
            requests[donor] = (AF3MSACache(repo, merged), donor_modality, donor_sequence, merged)

    ready, pending = {}, {}
    for key, request in requests.items():
        if key in aliases:
            continue
        cache, modality, sequence, settings = request
        hit = cache.lookup(modality, sequence)
        if hit is not None and not settings['refresh']:
            ready[key] = hit, True
        elif settings['mode'] == 'cache-only':
            ready[key] = cache.ensure(modality, sequence), True
        else:
            pending[key] = request
    if pending:
        capacity = available_cpus()
        budget = max(1, min(72, capacity - max(2, capacity // 6)))
        for _, _, _, settings in pending.values():
            if settings['cpu_budget'] != 'auto':
                budget = min(budget, settings['cpu_budget'])
        cost = max(max(1, query_cost(settings, modality)) for _, modality, _, settings in pending.values())
        if cost + 2 > budget:
            raise ValueError(f'MSA CPU budget {budget} cannot fit query cost {cost} plus 2 helper CPUs')
        workers = min(max(r[3]['search_workers'] for r in pending.values()), max(1, (budget - 2) // cost))
        def search(item):
            key, (cache, modality, sequence, settings) = item
            print(f"[msa] preparing {modality} context ({len(sequence)} residues, {settings['source']})", flush=True)
            return key, cache.ensure(modality, sequence)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for key, ref in pool.map(search, pending.items()):
                ready[key] = ref, False
    for key, donor in aliases.items():
        ref, hit = ready[donor]
        cache = requests[key][0]
        ready[key] = (None if ref is None else cache.reference(
            ref['entry_id'], ref['profile_id'], ref['generation_id'])), hit
    for key, (cache, modality, sequence, settings) in requests.items():
        ref, _ = ready[key]
        if ref is None:
            raise ValueError(f'Expected a ready {modality} context MSA reference')
        references.append(ref)
        saved_refs.append({'settings': settings, **{field: ref[field] for field in ('entry_id', 'profile_id', 'generation_id')}})
    for row in rows:
        key = row.pop('_request', None)
        row.pop('sequence', None)
        if key is not None:
            ref, hit = ready[key]
            row.update(status='cached' if hit else 'searched', cache_hit=hit,
                       **{field: ref.get(field) for field in ('entry_id', 'profile_id', 'generation_id', 'msa_depth', 'paired_msa_depth', 'template_count')})
    atomic_json(state, {'references': saved_refs, 'rows': rows})
    if rows:
        write_csv(Path(cfg['output_dir']) / 'msa.csv', rows)
    return references, rows
