# Cluster successful original backbones

`partidesign cluster` is independent CPU postprocessing. It uses validation's
success tables to select original generated structures, extracts their design
polymer chains, and clusters their backbone geometry. It does not generate
sequences, refold, relax, score with PartiScore, or load model weights.

```bash
partidesign cluster inference/protein/pepdbench --plan
partidesign cluster inference/protein/pepdbench
partidesign cluster TASK/validate/af2_ig/success
partidesign cluster TASK --method esmfold2-standard \
  --criteria complexa_ligand_binder --tm-threshold 0.5
```

`python -m src.cluster` accepts the same arguments. Multiple input paths are
allowed; repeated/overlapping paths identify one job only. Inputs may be a run
whose direct children are tasks, a task containing `result.csv`, a validation
method directory, or that method's `success` directory. This is a bounded
directory discovery, not a recursive scan of every structure under a tree.
`--task`, `--method`, and `--criteria` are repeatable selectors.

## Selection and coordinates

Each `TASK/validate/METHOD/success.csv` is one directory job. Each row identifies
a successful sequence slot; its `passing_criteria` and criterion decision fields
identify the standards it passed. Within each criterion, a backbone enters once
if any of its sequence slots passed. Original, FI, and ordinary MPNN evidence is
retained in the output's `passing_branches` field. FI is identified by the saved
fixed-interface settings, including when a branch has a custom name.

Tasks, validation methods, and criteria remain separate clustering cohorts.
There is no implicit cross-method success union/intersection. Backbone identities
are the task plus its final filename stem; search candidates sharing a
`base_sample_id` are still distinct structures.

The selected task's current `result.csv` supplies the final filename and
`design_chains` metadata. Saved absolute `input_file` paths may be stale after a
task has moved. Resolving the same sample against the current task recovers the
original structure without depending on those paths. A copied success directory
without its task/result metadata is insufficient; ambiguous or missing mappings
are errors rather than guesses.

Clustering uses the original structure referenced by the current result row,
including a retained relaxed structure when it is the published output.
The success decisions are the persisted validation decisions. If original
coordinates or the intended criteria change, refresh validation/metrics and
filtering before clustering again.
Successful refold files in `success/` are not extra clustering inputs. This
measures geometry of successful generated backbones, not predictor-dependent
refold conformations.

Design chain IDs are final CIF `label_asym_id` values stored as
`design_chains[].chain`. Author IDs do not override these labels. Protein traces
use CA; RNA and DNA traces use C3'. Context chains are excluded. Modified polymer
anchors remain included even if their original records are HETATM. Side chains,
crosslinker chemistry, and cyclic bonds are not separate features of this trace
comparison. Different design-polymer compositions are separate cohorts; lengths
within the same composition may differ. Multiple design chains remain distinct
and use US-align's multimer alignment.

## Score and clustering rule

US-align runs once per required unordered structure pair at its standard
alignment setting. Its two TM-scores, normalized by the two input lengths, are
combined as `(TM_L1 + TM_L2) / 2`. This is not normalization by the average length
(`-a T`). Protein and nucleic acids use US-align's corresponding scoring modes;
RNA and DNA remain separate application cohorts.

The greedy algorithm follows the public PXDesignBench binder postprocessor:

1. Among unassigned structures, choose the center with the most unassigned
   neighbors at or above the threshold, counting itself.
2. Assign the center and those neighbors to one cluster.
3. Repeat until all structures are assigned.

Ties follow natural sample/filename order, so `sample2` precedes `sample10`.
The first member is the geometric center. Members must pass against the center;
they need not all pass against one another. Threshold equality passes.
`--tm-threshold` defaults to 0.5 and accepts finite values in `(0,1]`.
Nucleic-acid and short-peptide threshold interpretation should be assessed for
the intended benchmark; the shared default is not a claim that its scientific
meaning is identical for all polymers.

Reference algorithm:
[PXDesignBench postprocess_binder.py](https://github.com/bytedance/PXDesignBench/blob/633f5d156b0c507fdc30112b21a42e4c99a77ebb/pxdbench/scripts/postprocess_binder.py).
The public script accepts an arbitrary input directory; using successful
original backbones is this project's explicit protocol.

## Scheduling and reports

The default concurrency is 16, bounded by available CPUs and the number of
directory jobs. `--workers N` accepts 1 through 16. Each process handles one
success directory at a time, limits CPU math threads to one, and evaluates its
structure pairs sequentially. No nested alignment process pool is created.
Within one directory job, criteria reuse each structure extraction and each
required pair score. Pair scores are recalculated from the current structures
on a subsequent command invocation.

`--plan` prints discovered jobs, distinct successful backbone counts, criteria,
and required comparison counts. It performs no structure alignment, tool lookup,
model work, or report/cache writes.

Selectors use resolved physical task/method directory names; overlapping paths
and symlink aliases submit the same physical directory only once.

Each completed method directory receives:

- `clusters.csv`: one member row per criterion/composition, with its cluster,
  center, cluster size, TM-score to center, successful branches, and original
  structure paths.
- `cluster_summary.csv`: successful backbone, cluster, singleton, and largest
  cluster counts per criterion/composition.

Reports record the threshold, score normalization, algorithm, coordinate source,
and US-align executable/version. Cluster IDs are local to a criterion and design
composition. No sample filenames, result rows, success links, or score ranks are
renamed. Empty inputs produce header-only reports (or zero-count rows for an
explicit empty criterion). A zero successful-backbone count is not a validation
failure rate: prediction may be unfinished, unavailable, or unfiltered.

Each directory's reports are prepared only after all its required alignments
succeed, then each CSV is replaced atomically. Ordinary publishing errors restore
already replaced reports from temporary backups; an unsuccessful restoration is
reported explicitly. A failed directory makes the command return nonzero while
other directory jobs can finish. Temporary traces, staged CSVs, and backups live
under the invocation directory's `.cache/cluster` and are removed on completion. When outputs are
on another filesystem, report staging uses a private `.cache/cluster` under the
method directory so replacement remains atomic. Run from the
repository root to keep them in the ignored cache.

## US-align installation

The Conda environment includes `bioconda::usalign`. Existing environments can use:

```bash
conda install -n partidesign -c conda-forge -c bioconda usalign
```

Discovery prefers `USalign` beside the running Python, then PATH. Override it
with `--usalign /path/to/USalign`. Conda packages can lag upstream; use a recent
official build when multimer alignment fixes are needed. The upstream build is
documented in [US-align](https://github.com/pylelab/USalign). This implementation
was exercised with upstream US-align 20261012; the executable remains an external
CPU tool rather than a bundled model or runtime download.
