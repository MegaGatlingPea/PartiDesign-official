"""Prepare context alignments from native task sources without sampling or models."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import yaml


def task_queries(cfg):
    from src.design.core.tasks import parse_task
    from src.design.core.compiler import prepare_task_source
    from src.design.rewards.af3_input import build_af3_input
    from .af3_msa import msa_query_enabled

    rows = []
    for name in cfg['task_names']:
        task = parse_task(name, cfg['task_payload'][name], invocation_dir=Path.cwd())
        prepared = prepare_task_source(task)
        if prepared.partial_reference is not None:
            record = prepared.partial_reference.record
            atoms = record.atoms
            context = ~np.isin(atoms.chain_id, record.design_chains)
        elif prepared.source is not None:
            atoms = prepared.source
            context = ~np.asarray(atoms.is_design, dtype=bool)
        else:
            continue
        context &= np.isin(atoms.mol_type, ['protein', 'rna', 'dna'])
        if not context.any():
            continue
        payload = build_af3_input({'atoms': atoms[context], 'design_chains': {}}, {},
                                  seed=cfg['seed'], protocol='complex').payload()
        configured = cfg['tasks'][name]
        policies = {}
        for stage in [*configured['rounds'], *configured['final'].get('validators', [configured['final']])]:
            if stage['method'] in {'af3', 'mol1'}:
                settings = stage['validation'][stage['method']]['msa']
                policies.setdefault(tuple(sorted(settings.items())), settings)
        if not policies:
            settings = configured['msa']
            policies[tuple(sorted(settings.items()))] = settings
        for chain in payload['chains']:
            modality = chain['modality']
            if modality not in {'protein', 'rna', 'dna'}:
                continue
            for settings in policies.values():
                rows.append({'task': name, 'chain': chain['source_id'], 'modality': modality,
                             'sequence': chain['sequence'], 'length': len(chain['sequence']),
                             'source': settings['source'],
                             'enabled': msa_query_enabled(settings, modality),
                             'settings': copy.deepcopy(settings)})
    return rows


def run(args, repo_root):
    from src.pipeline.config import build_config
    from .af3_msa import AF3MSACache, resolve_msa_settings
    from .msa_options import query_identity

    cfg = build_config(args, repo_root)
    rows = task_queries(cfg)
    caches, requests, prepared, results, view = {}, {}, [], {}, []
    searches = 0
    for row in rows:
        settings = resolve_msa_settings(row.pop('settings'), repo_root)
        cache_key = tuple(sorted(settings.items()))
        if cache_key not in caches:
            caches[cache_key] = AF3MSACache(repo_root, settings, readonly=bool(args.plan))
        cache = caches[cache_key]
        query = query_identity(settings, row['modality'], row['sequence'])
        prepared.append((row, settings, cache, query))
        if row['enabled']:
            previous = requests.get(query)
            if previous is None or (settings['mode'] == 'auto' and previous[0]['mode'] != 'auto'):
                requests[query] = settings, cache
            elif settings['refresh']:
                settings = dict(previous[0], refresh=True)
                requests[query] = settings, AF3MSACache(repo_root, settings, readonly=bool(args.plan))
    for row, settings, cache, query in prepared:
        row.update(cache_dir=str(cache.cache_dir), templates=bool(row['enabled'] and row['modality'] == 'protein' and settings['templates']))
        if not row['enabled']:
            row['status'] = 'query-only'
            view.append(row)
            continue
        if query not in results:
            settings, cache = requests[query]
            cached = cache.lookup(row['modality'], row['sequence'])
            if not args.plan and settings['mode'] == 'auto' and (cached is None or settings['refresh']):
                searches += 1
            if not args.plan:
                print(f"[msa] {row['task']}:{row['chain']} {row['source']} {row['modality']} L={row['length']} "
                      + ('refreshing' if settings['refresh'] else 'cache hit' if cached else
                         'searching MSA + protein templates' if row['modality'] == 'protein' else 'searching RNA MSA'),
                      file=sys.stderr, flush=True)
            reference = cached if args.plan else cache.ensure(row['modality'], row['sequence'])
            hit = cached is not None and not settings['refresh']
            status = ('refresh-required' if args.plan and settings['refresh'] else 'cached' if hit else
                      'search-required' if args.plan else 'ready')
            results[query] = reference, hit, status
        reference, hit, status = results[query]
        row.update(status=status, cache_hit=hit)
        if reference:
            row.update(entry_id=reference['entry_id'], profile_id=reference['profile_id'],
                       generation_id=reference['generation_id'],
                       unpaired_msa=reference['unpaired_msa_path'],
                       template_count=len(reference.get('templates', [])))
        view.append(row)
    print(yaml.safe_dump({'cache_policy': 'shared by AF3 and Mol-1', 'queries': view,
                          'unique_queries': len(results), 'search_started': searches > 0},
                         sort_keys=False), end='')
    return 0
