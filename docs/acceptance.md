# Inference repository acceptance

Status: accepted, 2026-10-10. No remote publication.

The reference is original PartiDesign main c997b40. Implementation and acceptance
use Ubuntu-jzc `/home/megagatlingpea/workdir/partidesign-inference`;
the original checkout and environments remain unchanged.

## Release boundary

Business directories are `src/`, `configs/`, `ckpts/`, `community_models/`,
`docs/`, `licenses/`, and `partidesign/`. Private tests, builds, outputs and
runtime filesystem caches belong in ignored `.cache/`. No Protenix namespace,
training datasets, root examples/benchmarks/resources, or `configs/inference/`
directory is shipped. Configs are flat; examples and benchmarks are under
`configs/examples/` and `configs/benchmarks/`. Native chemistry has one
`configs/resources/chemistry/ccd.json.gz` containing 48,964 components.

## Verified behavior

- Full final-source CPU regression in the fresh environment: 1,265 passed, 32 skipped. Hardware/opt-in skips are
  supplemented by the real native/backend matrix below; they are not counted
  as successful hardware execution.
- Six routed native members generated structures with the production 400-step
  solver; all eleven retained MPNN/scoring/refold adapter smoke cases succeeded.
  The final installed wheel separately passed the full 17-case matrix in the
  fresh release environment, using an explicit external model root.
- Direct checkpoint loading, DNA generation, PartiScore search, fold reward,
  AF2 cyclic GT+Soluble validation, AF3/AF3-LA public validation, RNA GT+NA-MPNN
  validation and antibody scoring succeeded.
- Default compiled CUDA generation succeeded. Protein, cyclic peptide, RNA and
  ligand OpenMM relaxation completed 4/4 on CPU and 4/4 on CUDA through Ray.
- All five MPNN families made three successful requests in one owner with one
  weight load. ESMFold2-fast, AF3 and AF3-LA each initialized once across three
  successful predictions. AF2 model1 initialized one complex and one monomer
  model, then reused both across three calls/six successful predictions.
- Four matched PartiScore input kinds produced exactly equal scores to the
  reference. All 76 tensors match except the intentionally repaired centering
  affine: the reference incorrectly replaced its translation with identity.
  The new affine passes source/centered coordinate round-trip checks.
- Five alternating fresh-process timing pairs and five warm calls per input
  passed the 1.10 ratio limit for preparation, network forward and full scoring.
  Largest cold preparation ratio is 1.0544; largest warm preparation ratio is
  1.0206. Native controlled cold ratio is 0.9444, warm generation 0.9848 and
  CIF writing 0.9633.
- The final YAML created a fresh release environment in one successful conda
  invocation after correcting proxy propagation. Exact-version/import and
  installed-package version/import and `pip check` verification passed.

## Distribution and corrected source behavior

Editable installation, wheel build/install, source distribution, and a wheel
built from that source distribution all passed. Final archives contain the two
root legal texts, exactly one native CCD, and the required ColabDesign package
initializer. They contain no Protenix, native training datasets, private tests
or caches. The final installed wheel produced all six native branches and ran
all eleven backend smoke cases: 17/17 passed.

All 647 retained benchmark tasks and 46 example tasks compiled (693 total,
one representative per task). Free-ligand cases resolve a length using the
runtime distribution; trusted-CCD cases were checked with a saved, guarded
production-spawn harness. Initial harness failures remain in private logs.

An actual Git clone preserved the seven business directories and three curated
Git weight assets. Its six native strict CPU loads, public/module CLI and
PartiScore forward all passed. Imports came from the clone; large external
weights were selected explicitly with `--model-dir`.

Final-wheel Vina validation completed with finite affinity (-6.124 kcal/mol)
and valid chemistry; that example did not pass its scientific success preset.
The supported DockQ PDB API executed with a finite interface score (1.0 for a
self-comparison); no root DockQ checkout was created.

Corrected original errors include the scorer centering affine and modality-aware
validation failure records. Closed fixed modified-polymer products use their
existing whole-context force-field route. A generic CCD valence proof prevents
PDBFixer from restoring a leaving atom consumed by an authoritative product
link; ambiguous/incomplete chemistry still fails explicitly. The linked-case
CPU output preserved all 201 public atoms and edges, both crosslinks and all
58 fixed coordinates exactly. Its unknown source bond orders were explicitly
resolved from the retained task's authoring rules, never guessed.

Worker import paths use only release-owned code/SDK directories. Poisoned
caller paths cannot load Protenix or the old training source. No per-request
model reload, weight freeze or numerical sampling change was introduced.

## Final review

Exactly three independent read-only reviewers reviewed the frozen source.
Correctness, source-boundary and packaging reviews all passed after repaired
findings and same-cohort re-review. Local main contains the implementation
and final acceptance report. The final index/worktree is clean; no new remote
was created and nothing was pushed.

Raw evidence is private: `.cache/acceptance/`, `.cache/final-smoke/`,
`.cache/final-workflows/`, `.cache/final-residency/`,
`.cache/final-score-performance/scatter-repair/`, `.cache/final-compiled/`,
`.cache/final-package/`, `.cache/final-linked/` and the external
`/home/megagatlingpea/workdir/.goal-artifacts/partidesign-inference/` directory.
These logs, tests, builds, structures and review snapshots are not Git content.

Large native/refold weights remain external assets under the selected model
root. Public bundle download URLs and a new GitHub remote are separate
publication prerequisites; neither is invented or silently created here.

## Static resource relocation, 2026-10-10

All 47 runtime data files formerly stored under `src/design` now live in
`configs/resources/{ligand,model,antibody,openmm}`. This includes the unchanged
ligand atom-count distribution, AE architecture, 21 paired antibody YAML/CIF
scaffolds, UFF table and adjacent licenses. All 47 files match the previous Git
revision byte-for-byte; the old source locations and empty asset directories
are removed. Native CCD remains the existing single chemistry resource.

The four consumers resolve data through the existing `resource_root()` helper.
Checkout/editable installs read `configs/resources`; wheels read the same tree
inside `partidesign/_resources`. Changing the working directory or selecting an
external model-weight directory does not redirect immutable data. Source and
wheel layouts retain no duplicate copy of these assets under `src`.

Fresh checks for this change:

- Focused scaffold, ligand, AE/bundle and coordination regression: 110 passed,
  1 unavailable-local-checkpoint skip.
- Wheel, source distribution and wheel-from-source-distribution: 3 passed.
  The installed-layout probe reads all four asset kinds and loads all 21
  scaffolds from an unrelated working directory with an external weight root.
- Public CLI from an unrelated working directory and strict CPU loading of all
  six native checkpoint branches: passed.
- All 21 relative YAML-to-CIF references, the 47 direct byte comparisons and
  `git diff --check`: passed.
- The tested Python 3.12 release environment received a fresh editable install.

This follow-up changes resource locations and packaging only. At the time of this resource-only change, the old named `partidesign`
environment was Python 3.11 with CPU PyTorch. The CLI goal subsequently
replaced it with the accepted Python 3.12 environment, as described below.

## CLI experience and named environment, 2026-10-10

The maintained environment is `/mnt/software/miniforge3/envs/partidesign`.
`conda activate partidesign` resolves the installed editable CLI to this checkout.
It preserves Python3.12.6, Torch2.9.1/cu128 and the accepted runtime versions;
`pip check`, all six strict native CPU loads and the native AF3 CIF SDK pass.
The temporary release/acceptance environment prefixes were removed after these
checks. The driver mismatch observed during that earlier CLI check was separate from
repository acceptance. The filter cleanup below verifies CPU and CLI behavior; it
does not claim new folding-model GPU execution.

Docs now live directly under `docs/*.md` and `docs/tasks/`. The configured default generation
output is `./inference/${run_name}`; help/quickstart/validation paths match
that default. Explicit `--out` examples choose their own output directories.

Terminal presentation uses adaptive Rich views, with a prompt-toolkit3.0.53
checkbox dialog only for no-target downloads in a real terminal. Existing explicit
and piped calls keep deterministic noninteractive behavior. Presence, URL origin,
actual stage results and skipped optional resources remain separate. Read-only
inventory and cancellation do not initialize model-root caches.

ProtDBench/PepDBench each have base, coarse-alpha and coarse-beta task maps.
All56 new task entries compiled on CPU with exact alpha1/beta2 design-chain masks,
unchanged budgets and preserved sequence/bond/cyclic-offset constraints.
Head-tail cyclic PepDBench has its own CyclicMPNN validation profile at0.0001;
linear/disulfide uses ProteinMPNN at0.0001. Both preserve original1/FI1/MPNN4,
FI6A, AF2-IG model1,3recycles and the peptide criterion.

The final CLI, PTY, packaging and independent-review outcome is recorded in
ignored `.cache/goals/cli-experience/state.md`. Private tests, intermediate builds
and terminal previews remain outside the public distribution.


## Shared criteria cleanup, 2026-10-10

Acceptance checkout: Ubuntu-jzc `/home/megagatlingpea/workdir/partidesign-inference`,
using `/mnt/software/miniforge3/envs/partidesign/bin/python`. Changes remain
uncommitted and unpushed at the user's request.

- 17 criteria, 15 design validation profiles, unchanged four AF2 presets/backend;
  all `af3_*` families support AF3, AF3-LA, ESMFold2-fast and standard.
- All 660 manifest entries and sampling inputs/tasks remain unchanged. MotifBench
  retains 30 sampling tasks with external official scoring. Docking suites
  (`nuc_ligand`, PoseBusters, ssDNA/ssRNA) retain sampling and have no internal
  validate profiles. The requested rollback removes the added RMSD module,
  method, CPU scheduler branch and CLI options completely.
- Protein/NA monomers have separate profiles/CLI selectors, and DNA binder
  redesign uses NA-MPNN. De novo ligand design retains ODesign/Vina or optional
  native iPTM screening; docking has no refold success criterion.
- Generic Complexa AME is retained. Ligand binder alignment uses strict chemical
  correspondence, a complete binder heavy fit or CA fallback, and noncollinear
  anchors; target-to-binder PAE remains separate from bidirectional diagnostics.
- Shared antibody geometry supports every antigen modality. Pinned BoltzGen,
  BindCraft2 and RFAntibody comparisons identify metric masks, normalization and
  reductions. The 10-Angstrom/2-Angstrom antibody screen remains explicitly
  provisional and RF2-inspired; no new confidence threshold/refold is added.
- Before the subsequently cancelled docking extension, full focused validation
  passed 529 tests (12 skipped), and CLI collections passed 80 tests. The final
  rollback regression also passed **609 tests, 12 skipped** (81.34 s), with no
  failures. Three independent
  reviews and repair checks covered the retained runtime/geometry/config changes.
- Public CPU checks compile Fab/VHH examples and nine antibody benchmark tasks,
  one seeded representative each. This does not certify all 1,800 generated
  samples. Four-method `validate --plan` selects actual fast/standard mode;
  `design --show-config` and help use `./inference/<run_name>`.

No folding-model GPU run or new success-rate measurement is claimed. Sampling,
solver/compile settings and persistent model reuse remain unchanged. Existing
editable installation reads source/config changes; no reinstall/dependency
change is needed. Test evidence stays in ignored `.cache/acceptance/filter-cleanup/`.

## MPNN sampling repair, 2026-10-11

Acceptance uses the Ubuntu-jzc inference checkout and its `partidesign`
environment. This update repairs sequence sampling and its callers; FM inference
seeds, solver settings, temperatures, routing and existing H100 runs are unchanged.

The ordinary and antibody ProteinMPNN grouping bug compared candidate-specific
JSON keys as constraints. Equivalent copies were consequently sampled separately
with the same seed. Groups now compare constraint content and serialize the
selected target key only when writing native input. Soluble and CA-only paths
share the correction. Genuinely different constraints remain separate.

Additional repairs preserve partial-resume draw indices, restore paired-antibody
CSV roles/mutable regions, honor user-fixed positions in derived antibody plans,
forward PartiScore fixed-interface settings, reject duplicate candidate IDs,
provide independent NA batch decoding-order noise, and preserve cyclic retry
keys. CPU model and packer initialization no longer shifts the first sampling
stream; request-only omission does not create another cyclic resident model.

Verified final source:

- Four focused model-free repair files: **93 passed**. The existing validation
  regression suite: **529 passed, 12 skipped**. Skips are not hardware passes.
  The original source fails the targeted bug-reproduction cohort in 24 cases;
  it was loaded privately without resetting the checkout.
- On the RTX 4080 SUPER, all five public MPNN sources pass count 1/4/8,
  fixed/context preservation, default protein C omission, same-seed replay,
  explicit seed zero, partial resume and already-complete skipping. Each source
  makes nine native requests in one worker with load counts `[1,0,0,0,0,0,0,0,0]`.
- Real BHRF1 and MDM2 backbone inputs reproduce their ordered eight draws across
  three runs. They produce respectively 8 and 5 distinct sequences at the
  unchanged ProteinMPNN temperature `0.0001`. This is evidence for these inputs,
  not a guarantee of eight unique sequences at low temperature.
- Native CA-only GPU sampling passes two calls. Paired VH/VL ProteinMPNN,
  SolubleMPNN and LigandMPNN each pass two eight-draw calls, protecting FR and
  changing only CSV-declared mutable positions; each cohort has 8 distinct
  sequences. Ligand side-chain packing passes two additional GPU calls and
  produces the expected eight packed structures.
- Direct native CPU Protein, Soluble, Ligand and Ligand-plus-packer calls reproduce
  cold/hot sequences, with load counts `[1,0]` or `[2,0]` for main model plus
  packer. These exercise the CPU scientific runners; the public owner adapter
  continues to require an assigned GPU and explicitly rejects a hidden-GPU
  environment.
- A matched GPU BHRF1 native batch of eight compares the original and repaired
  resident worker in original/current/current/original order. Eight warm calls
  per version produce equal sequences. Median time is 0.7782 s original and
  0.7757 s repaired (ratio **0.9967**, acceptance limit 1.10). This checks the
  resident sampling path; it is not a whole-benchmark throughput measurement.
- Exactly three independent read-only reviewers pass correctness, RNG/cache and
  caller integration checks. Their CPU tests and source inspections supplement
  the main thread's real-model checks. No refold GPU accuracy claim is made.

The persistent model lifecycle remains intact. No dependencies, package layout
or install metadata changed, so the existing editable installation reads these
repairs without reinstalling. Start a new validation owner to load the updated
worker code. Completed pre-fix slots remain completed until an explicit `--rerun`;
see [MPNN sampling and resume](mpnn.md#sampling-batches-seeds-and-resume).

Evidence and test scripts remain ignored under
`.cache/goals/mpnn-sampling-20261011/` and `.cache/tests/`. They add no public
fixtures, model copies or hashes to the release. Benchmark success rates must be
recomputed from corrected validation runs; this acceptance does not rerun the
H100 production benchmarks or certify TNFA clash quality.


## 2026-10-15 — successful-backbone clustering

Added the independent CPU `partidesign cluster` / `python -m src.cluster`
command. It reads validation success decisions, resolves original generated
structures against the current task/result table, extracts design polymer
traces, and uses US-align's two length-normalized scores with the public
PXDesignBench greedy-center rule. Methods, criteria, and polymer compositions
remain separate; one single-thread worker handles one success directory, with
a concurrency limit of 16.

Actual acceptance on the final implementation:

- Ubuntu-jzc: **146 focused clustering/CLI tests passed**. pro5k: **121 focused
  clustering tests passed**. These include current-path recovery after moving
  task trees, multiple methods/criteria, sequence-slot deduplication, independent
  search outputs, symlink aliases, stable ties, exact threshold boundaries,
  report-publishing rollback, and directory process scheduling.
- Real US-align **20261012**, compiled from official commit
  `746ad8a2a4dea9f317075cc482df0ecb5440bdea`, ran on both hosts.
  CPU fixtures cover protein and DNA monomers, protein/DNA/mixed multimers,
  total design-chain lengths and context exclusion; an additional end-to-end
  RNA/DNA fixture verifies separate cohorts.
- pro5k PepDBench normal/alpha/beta completed **54/54 directories**, with
  **2,907 successful original backbones** and **289,946 unordered comparisons**.
  The final run used **16 worker PIDs**; a live process snapshot confirmed
  **one thread per worker**. Twelve empty success cohorts produce zero-count
  summaries.
- Independent inspection checked all **108 reports**, all **2,907 member rows**
  and all **235 clusters** against the original success and result tables.
  Centers, cluster sizes, representative paths and center TM-score thresholds
  agree. Four pair scores from two small KEAP1 cohorts were independently
  recomputed directly from original CIFs and matched the published clusters.
- A built wheel was installed into an isolated ignored target directory.
  From outside the checkout root it completed a real two-backbone PepDBench
  clustering job. The wheel's cluster modules match the source and that
  command imports neither Torch nor JAX.

| PepDBench run | Successful backbones | Clusters at mean TM >= 0.5 | Singleton clusters |
| --- | ---: | ---: | ---: |
| normal | 984 | 85 | 45 |
| alpha | 938 | 76 | 38 |
| beta | 985 | 74 | 37 |
| total | 2,907 | 235 | 120 |

Scope and limitations:

- These are real CPU structural-alignment and report-consistency results.
  No new sampling, MPNN generation, refolding, or GPU scientific-metric
  acceptance was performed. RNA/DNA threshold calibration and multimer
  diversity on a full biological benchmark remain unverified.
- Three cached legacy tests in `test_cli_model_root.py` initially failed,
  including with the CLI loaded privately from original commit `0fab536`.
  The cause was fixture placement: an invocation simulated as outside a checkout
  was actually below this checkout's ignored `.cache/`, so normal ancestor
  discovery found the real repository before reaching the mocked code root.
  This was not an installed-source registry defect. The test fixture now isolates
  the real checkout while retaining synthetic checkout discovery; all six model
  root tests pass. No production model-root behavior was changed.
- Evidence, temporary structures, the wheel and tests are ignored under
  `.cache/cluster_implementation/` and `.cache/tests/`. Formal PepDBench
  `clusters.csv` / `cluster_summary.csv` are beside each method's success
  table. Input structures, result tables, filenames, score ranks and persistent
  model caches are unchanged.


## Notebook task-authoring prototype (2026-10-15)

The following records describe the earlier prototype. Checks for the revised
workflow are recorded in the final section below.

Ubuntu-jzc, formal `/mnt/software/miniforge3/envs/partidesign/bin/python`:

- Installed ipymolstar 0.1.0, anywidget 0.11.0, ipywidgets 8.1.9,
  nbformat 5.10.4 and nbclient 0.10.2.
- Executed all nine notebook cells in a fresh kernel; no automatic structure
  download, model loading or sampling. Source notebook has no saved outputs.
- Final backend CPU suite: 18 passed in 26.70 s; five focused condition-color
  checks also passed. Includes real RCSB 1CRN entry/assembly fetch, native source graph round trips,
  representative compiler requests, partial motifs, coordinate-only ligands and
  new-chain ATP CCD chemistry from the existing local asset (CPU only).
- Programmatic widget smoke and six integration scenarios passed: source
  normalization, click-to-mark, exact input preview, partial-atom contig,
  de novo counts, YAML export and actual CLI `--show-config`, result-view
  selection isolation, categorical-color payloads and fresh-kernel execution.
- Eight existing writer checks passed. The new B column was the only CIF data
  change versus HEAD; native coordinate-only refinement preserved it.
- On one 1,567-atom / 1,591-bond CIF, alternating five writes gave medians
  35.407 ms before and 36.037 ms after; run ranges overlapped. This is a small
  CPU observation, not a production throughput guarantee.

No browser rendering/click round trip, sampling, GPU, OpenMM minimization,
refolding or scientific-quality acceptance was run. Antibody-specific notebook
forms/previews are outside this prototype. Legacy structures without condition
B values are not backfilled. Tests, executed notebooks and temporary inputs
are under ignored `.cache/notebook_v1/`.

`pip check` still reports the pre-existing Lightning 2.5.0 requirement for
packaging <25 against installed packaging 26.3; the notebook install did not
change packaging. This is not a clean-environment installation gate.


### 中文界面与模态联动（2026-10-15）

在同一 Ubuntu-jzc 正式 partidesign 环境完成以下 CPU 验收：

- 界面模态矩阵 29 项通过（23.09 s），另有多字符链 SS 示例的 YAML
  解析检查通过；已有界面冒烟脚本通过。
- 中文后端回归 18 项通过（23.10 s）；后端本轮仅修改显示文案。
- 主线程 9 组集成场景通过：五模态的合法 coarse SS 原生编译、旧条件
  清理、非法 coarse loop 拒绝、蛋白与 RNA token/pair 表单编译、手写
  蛋白/RNA 联合任务跨模态保留及导出、源文件变更保留手改文本、
  backbone 源链与长度、RNA 部分原子基序、中文笔记本全新内核整本执行。
- 笔记本保留 9 个单元（4 个代码单元），源码不保存执行输出。
- 文档中错误的 coarse loop 说明已按原生编译器修正；核酸 token 标签
  也仅接受 paired/unpaired。没有修改原生 SS 编译规则。

浏览器实际渲染/点击仍未实测。Mol* 自带菜单及上游异常可以保留英文；
本轮无模型采样、GPU 或科学质量验收。测试与执行结果仍仅在忽略的
`.cache/notebook_v1/` 中。

### 使用者工作台：笔记本与文档检查（2026-10-15）

- 当前笔记本为 6 个单元、2 个实际交互代码单元：主编辑器和独立高级区。
  nbformat 校验及全部代码单元的 Python 语法检查通过；源码不含执行输出。
- 文案与入口检查通过：Input／模态先行，Motif 逐残基原子选择及显式
  contig，Context-only Hotspot，高级条件及完整 YAML 的显式应用。
  文案明确源 CIF 已有原子的默认保留、清空限制恢复全部、Mol* 联动高亮
  以及独立的待选／已应用状态；此处只核对文案，不代表交互验收。
- `git check-ignore --stdin` 确认 `configs/customs/raw/example.pdb`、
  `configs/customs/input/example.cif` 和 `configs/customs/tasks/example.yaml`
  均被根目录的 `/configs/customs/` 规则忽略。
- 以上是笔记本格式、代码语法、文档及忽略规则检查；不代表界面点击、
  原生任务编译、采样或科学指标验收。检查记录保存在忽略的
  `.cache/notebook_v2/docs_validation.json`。


### 使用者工作台：CPU 与真实浏览器验收（2026-10-15）

- 当前版本的合并 CPU 验收为 **99 passed in 28.56 s**，无跳过或 xfail：
  UI 39、Motif/输入辅助函数 47、配体边界 7、打包 4、Hotspot 清空和高级
  预览位置 2。蛋白、RNA、DNA、配体的任务均通过原生 CPU 编译检查；
  没有用模拟模型输出替代编译结果。
- 真实 Chrome/JupyterLab 验收完成：从文件导入、Mol* 序列及三维点击、
  Hotspot 应用、Motif 逐原子设置、Contig 长度更新、条件预览和任务导出。
  三维 CA 点选实际产生 `atom_id: ["CA"]`，只保留 CA；再次点击后恢复
  该 VAL 残基全部 7 个源原子。RNA A2 的局部预设提供 15 个实际解析的
  原子，`4/A2/5` 经原生检查后在独立高级区按 B=80 显示。
- 最终默认布局经全新内核实测：一个 Mol* 实例，序列面板可见、侧边工具
  默认收起，容器内布局；880 px 宽窗口表单与查看器上下排列。最终浏览器
  无页面错误。加载完成前的配色/高亮请求被排队，重复挂载释放旧监听和
  查看器；未改安装的 ipymolstar 包。
- 配体 atom_motif 的原子预算等于保留原子数时，原生编译器直接使用
  Motif，不再创建零原子 remainder。自由配体的非正数量仍被拒绝。
- 原始文件重名不会覆盖不同输入；完整 Motif 来源链保留在规范化 Input。
  改动后旧预览、检查结果和命令失效，导出使用当前任务。清空最后一个
  Hotspot 后也可应用空集合。
- 真实 JS 资源已通过实际源码闭包复制检查；打包规则保留该资源并排除
  整个 configs/customs，且没有误排除其他同名子目录。未构建完整 wheel。
- 当前源码笔记本为 6 个单元、2 个代码单元；nbformat/语法通过且已清除
  执行输出。自定义任务在 configs/customs 的 raw/input/tasks 下保存并
  被 Git 忽略；测试、截图、导出验收文件和日志均在 .cache 下。

这是任务准备、编译和交互验收；没有运行模型采样、GPU 推理、refold、
OpenMM 优化或科学质量评测。没有更改推理循环、persistent model cache
或本轮的 dump 写入路径。临时 Jupyter 服务和浏览器已关闭。CPU 记录为
`.cache/notebook_user_v2/final_acceptance.log`，浏览器记录为同目录
`browser_report.json`；较早原型的验收范围不代表当前界面。


### English notebook, Reference workflow, and InsulinR preset (2026-10-15)

- Combined CPU acceptance: **144 passed in 50.68 s**, no skips or xfail.
  Covers input/Contig/atom helpers, Reference contact discovery, widget state,
  preset loading, imported-YAML filter locking, native ligand boundary behavior,
  packaging rules, and preview placement. Tests and outputs are under the ignored
  `.cache/notebook_user_v3/`; no checkpoints are loaded by authoring checks.
- The product notebook has six cells, including two code cells, passes nbformat
  and Python syntax checks, and contains no saved execution output. It opens the
  editable local ProtDBench InsulinR preset. Original target hotspots E64/E88/E96
  map to normalized selectors E59/E83/E91 while preserving author identity. The
  native CPU check expands 81 lengths × 4 samples = 324 structures and checks
  eight representative plans; it does not start generation.
- Actual Chrome/JupyterLab interaction verified source loading, live chain
  filtering and restoration from raw input, separate Context/Reference roles,
  full-Reference contact candidates without a Motif, explicit Hotspot merge, and
  export excluding unused Reference. MolStar's hierarchy contained only A after
  applying the A-chain filter. Selecting B2 (author B20, SER) in the actual
  sequence viewer populated the Motif basket. The exported `10/B2/15` condition
  retained exactly CA/CB; its input retained the complete required source chain.
- The visible ligand **From distribution** selector passed the native CPU check
  using a real Context pocket. At an 880 px viewport the document width remained
  880 px; one MolStar viewer instance was present and the advanced preview
  remained usable. Base colors are muted; categorical B values are unchanged.
- A JupyterLab startup page error (`Invalid data cell.`) was observed in the
  fresh browser session. The widgets, MolStar source/condition views, native
  checks and exports worked; this is not a claim of zero browser console errors.
  The test server, kernel and browser were temporary and are closed after checks.

This verifies task preparation and CPU/browser behavior, not GPU sampling,
refold, OpenMM relaxation, or scientific quality. No inference loop, persistent
model cache, or runtime dump path was changed in this iteration. Evidence:
`.cache/notebook_user_v3/final_acceptance.log` and `browser_report.json`.


### Nord notebook themes and aligned options (2026-10-15)

- The workbench now offers synchronized sun/moon controls for Nord Light and
  Nord Dark in the basic and advanced headers. Native widget semantics and
  keyboard access remain; styling is scoped to `.pd-workbench` and the CSS is
  included in the source/wheel resource closure. Existing categorical B values
  remain unchanged.
- Final focused CPU acceptance: **51 passed in 25.65 s** (13 theme invariance
  checks plus 38 existing editor, packaging, filter and interaction checks).
  Switching does not call source loaders, previews or checkpoint loading; task
  YAML, checked-preview identity, command and input bytes remain unchanged.
- Actual Chrome/JupyterLab tests rotated the loaded InsulinR structure and added
  a real sequence-view Hotspot pick, then switched themes four times. Camera
  snapshots were identical, the same 1,237-atom structure remained, no additional
  loadComplete events occurred, and the four pending picks were retained. The
  supported MolStar canvas background setter updates display colors directly.
- At 1,580 / 1,200 / 880 / 680 px, visible fields stayed within their parents;
  all four step controls remained available. Both checkbox labels and inputs
  had zero vertical-center offset, and the option group wraps when needed.
  Expanded 36 px fields now have matching row heights; source tabs fit one row
  and the current native accordion and MolStar panel classes follow the theme.
  Keyboard Space also toggles themes. The native InsulinR task check still
  expands 324 structures and checks eight representative length plans.
- The final browser session had no page errors. CSS parser/scope and selected
  token contrast checks are recorded separately; they are not a complete
  accessibility certification. This is CPU and browser acceptance, with no GPU
  sampling, refold, relaxation or scientific-quality evaluation.

Evidence and temporary artifacts are under ignored `.cache/notebook_nord/`,
including `final_cpu.log`, `camera_report.json`, `responsive_report.json` and
Light/Dark screenshots. The test browser and service are closed after testing.


### Compact five-step task notebook and example presets (2026-10-15)

This iteration supersedes the four-step layout and Light/Dark toggle described
in the earlier notebook acceptance entries. It uses fixed Nord Light and one
shared five-step task frame: Input & design, Motif, Hotspot, Presets & advanced,
and Save. The form and Mol* cards are 520 px high; form contents scroll.

- Native backend acceptance: **54/54 example tasks passed**, plus **four source
  selector mapping gates**. Original and normalized source tasks were compiled
  on CPU and compared for masks, topology, secondary structure and sample
  counts. Preview and save/read checks passed, including eight antibody cases.
- Final widget acceptance: **61 passed in 108.03 s**, with no skips or expected
  failures. All 54 presets loaded, preserved parsed semantics and passed native
  CPU preview (86 representative plans and 270 planned requests, not generated
  structures). The checks also cover five-step state, help tooltips, empty-view
  state, 12 complete bond examples, and exact legacy RNA masks/coordinates/bonds.
  Checkpoint loading was forbidden during these checks; source example YAML
  bytes remained unchanged. Joint/antibody, legacy partial nucleotide motifs,
  and context-ligand cases use complete task YAML when the form cannot express
  the native task without losing conditions.
- Actual Chrome/JupyterLab acceptance checked all five steps at viewport widths
  1580, 1200, 880 and 680 px: **20/20 layout cases** retained 520 px cards, no
  horizontal workbench overflow, and no visible controls clipped by any
  scrolling ancestor. Wide layouts align both card edges; narrow layouts stack
  them. Source-tab margins no longer shift fields beyond their parent. Focus
  uses a complete 2 px inset ring instead of a clipped external outline.
- A new notebook with no preset showed the real Mol* canvas and 29 toolbar
  buttons with zero structures. Loading a source and clearing it retained the
  same viewer instance and restored an empty canvas. No dummy coordinates or
  placeholder structure download is used; the isolated empty-view browser
  session recorded zero page errors.
- Actual Mol* overpaint bundles before and after Check task contained exactly
  **33 Hotspot atoms on E59/E83/E91**, unchanged across the check. Position-based
  color queries no longer include component-name predicates that PDBe MolStar
  prioritizes over residue numbering. The previous extra five PHE residues
  are no longer colored. Loading the TBMB whole-task bond example through its
  visible controls passed CPU preview and displayed 681 supplied atoms.
- CSS parsing/scope checks passed for all 122 rules; selected text and focus
  contrast checks passed. This is not a full accessibility certification.
- The product notebook defaults to motif/motif_fixture and has five cells with
  one code cell. nbformat validation and code compilation passed. Only the
  obsolete separate advanced code cell was removed; every retained cell kept
  its pre-existing non-source fields and saved output unchanged.

Evidence is confined to ignored .cache/notebook_compact/: backend/acceptance.log,
backend/test_source_selector_mapping.py, editor_pytest_final_second.log,
browser_report.json, css_final.json, and setup/chemical/empty screenshots.
Temporary test browser, notebook kernels, server and tunnel are closed after
acceptance. No GPU sampling, refold, relaxation or scientific-quality evaluation
was performed. Inference dump paths and persistent model caches were unchanged.
No commit or push was made in this iteration.

### Six-step Runtime notebook and explicit execution (2026-10-15)

This iteration supersedes the previous five-step frame and complete-YAML-only
fallback for complex presets. The six steps end in **6 Runtime**, which owns
inference, validation, clustering, and opening results; the first five steps
retain source/condition previews and task export. Existing native preset fields
are editable visually, with Full task YAML available for structural edits.
Prepare buttons generate commands and frozen inputs. Execute and Modify command
default off; only enabling Execute and explicitly executing the command starts
an owned tmux job.

Current acceptance:

- **61 editor checks**, **20 Runtime checks**, and **24 backend checks** passed.
  The final main-thread Runtime rerun passed in **32.64 s**.
  They cover all 54 preset loads, native form preservation and edits, command
  preparation/execution guards, validation configuration, immutable input and
  custom scaffold snapshots, owned jobs, and result catalogue/preview wiring.
  **Four existing packaging checks** also passed.
- **120 actual-browser geometry cases** passed with no clipping.
- Actual-browser result preview passed with existing artifacts: **raw (122
  atoms) → AF3 (123 atoms) → AF3-LA (123 atoms)**, followed by a return to the
  Source preview. Result-opening controls were absent before **6 Runtime**.
  Mol* 3.3.2 used native `chain-id` coloring for refolds and retained confidence
  values; those values were not interpreted as task-condition categories.
- A real owned tmux job ran a harmless probe and exited **0**. Model launches in
  the Runtime widget checks were intercepted rather than executed.
- A dedicated read-only reviewer reported no remaining static blockers.
- Final Python compilation and diff checks passed. The copied setup source
  closure matched all six required modules/assets byte-for-byte. The isolated
  browser, notebook kernel, server/tunnel, and owned tmux probe were closed
  after the checks.

Evidence and private fixtures remain under ignored
`.cache/notebook_runtime_goal/`. This iteration performs no real GPU/model,
OpenMM relaxation, or scientific-quality acceptance. Earlier model/release
acceptance remains historical. No commit or push was made.

### Complete task presets and compact Motif editing (2026-10-15)

The current workflow selects the complete task preset in **1 Input & design**;
**4 Advanced conditions** edits the already loaded task. The notebook default is
`peptide/peptide_bind_protein_betahairpin`, using the existing 15-residue peptide
example with fixed Cys2/Cys14, beta-sheet segments 1–6 and 10–15, adjacent pairing,
and the SG–SG bond. Earlier preset placement and default examples above describe
historical versions.

Focused native Motif CPU acceptance:

- **23 focused tests passed in 15.85 s**, including exact roundtrips and native
  parsing for 17 complex presets, one mounted residue atom editor, inert list
  navigation/focus, protein atom and same-residue bond endpoint batch toggles,
  atomic rejection of unresolved atoms, nucleotide legacy/group restrictions,
  multiple-Contig navigation, and an empty Motif.
- The existing compatibility script independently passed **17 native preset
  roundtrips**, **seven isolated edits**, and **five actual CPU source previews**.
  These overlap the focused test coverage and are not an additional 17 presets.
- The typed-widget/default/layout compatibility check and Python compilation
  passed. Model checkpoint loading was forbidden in these checks.

Evidence is under ignored `.cache/notebook_scope_v1/native/`: `pytest.log`,
`roundtrips.log`, and `widgets.log`. Integrated browser selection, help overlays,
and complete-preset loading acceptance are recorded below. No new GPU/model execution or scientific-quality acceptance is
claimed here.


Integrated acceptance for the current revision:

- All **54 example tasks** passed native load/parse/CPU condition-preview checks
  (95.41 s). The complete-preset first-step placement loop also passed, and
  **all 54 tasks were then selected and loaded through the real notebook UI**.
  The Load action carries the visible selected task key in its own request;
  rapid selection followed by Load was tested against a stale backend dropdown.
- The tightened native Motif form passed **23 focused tests** (14.49 s), including
  17 lossless preset roundtrips. The source summary is one line, a single-chain
  Contig has no redundant chain picker, and Min/Max share one row. Both ordinary
  and native Motif forms mount only the focused residue's atom editor.
- **Ten normalized selection-routing checks**, the separate 54-preset placement
  loop, **19 existing Runtime checks**, and two focused read-only review passes
  passed. Mixed protein/non-Motif or unsupported nucleotide bond endpoints are
  rejected before any partial task mutation. Source/design chain renaming and
  legacy nucleotide atom subsets retain correct Motif coloring.
- Actual notebook browser checks covered **48 layout cases**: ordinary and
  native Motif tasks, all six steps, and 1680/1280/900/680 px viewport widths.
  Both cards remain 520 px high; controls do not cross horizontal clipping
  boundaries. Source caption, Select scope and Chain/Residue/Atom share one
  compact header with aligned vertical centers, including narrow layouts.
- Actual pointer clicks on a displayed atom and a displayed chemical bond
  reached the native Molstar event stream. Exact same-residue and cross-residue
  bond endpoint handling also passed the focused adapter tests. Scope changes
  set native chain/residue/element granularity and retain the same structure,
  camera position/target/up and zero additional coordinate loads after the
  initial camera animation finishes. Atom mode uses ball-and-stick and restores
  the previous representation when leaving Atom mode.
- Help was tested in JupyterLab with actual widget serialization: hover, focus,
  click pinning, Escape, independent tooltip IDs and clipping-safe body portals.
  The final workbench contained no widget JavaScript-error placeholders.
- The notebook default remains the supplied beta-hairpin disulfide task. Its
  code, metadata and saved-output preservation checks passed; only the requested
  default preset string and descriptive Markdown were changed. Execute, Modify
  command and compile remain false by default. All ten required source modules
  and browser resources were copied byte-for-byte by the setup source closure;
  Python compilation and whitespace checks passed.

Evidence remains under ignored `.cache/notebook_scope_v1/`, including
`preset-browser.json`, `layout-native.json`, `layout-simple.json`,
`scope-browser.json`, browser screenshots, and the native/selection/review logs.
Only the isolated test browser, server, kernel and tunnel are closed after
acceptance. No model prediction, GPU sampling, relaxation or scientific-quality
acceptance was performed in this iteration. No commit or push was made.

## Notebook deep usability acceptance (2026-10-15)

This iteration changes the notebook authoring and Runtime interface on
Ubuntu-jzc. Existing checkout changes, example YAML files, model/dump paths,
and persistent model-cache behavior were preserved. No benchmark was started.

- The notebook offers 46 non-antibody examples. Selecting an example loads the
  complete task automatically; filtering Category without changing the current
  example preserves edits. Native panels and an already displayed source are
  reused across steps. Help uses an immediate client portal with no inherited
  native title and cannot intercept navigation clicks.
- Modality-driven visibility now wins over form CSS. Only eligible ligand
  lengths offer From distribution. Relax exports replace the retained structure
  in place; the extra override checkbox is gone.
- Runtime has direct method checkboxes, AF2 model 1, optional project/custom
  criteria presets, and no duplicate output-root label. Prepare and Execute
  submit one snapshot of the visible settings. Failed preparation, changing a
  task/output, or loading a preset invalidates the previous executable command.
  Explicit custom directories remain intact; historical job refresh does not
  silently adopt a different task's output.

CPU acceptance on final source:

- 165 integrated tests passed in 148.76 s: 53 editor/preview cases, 14 state
  cases, 57 native-form cases, and 41 Runtime cases. All 46 supported examples
  were parsed, preserved and checked through representative CPU condition
  previews. State coverage includes all 46 auto-selections, clearing old
  conditions, same-key category filtering, stable native panels, zero redundant
  source-data notifications, relax replacement and stale-command invalidation.
- 19 additional atomic-action cases passed in 2.86 s. They cover selection
  indices, current option labels and model identities, complete prevalidation,
  duplicate requests, acknowledgement/error handling, command edits and explicit
  execution. The actual JS snapshot helper also passed its separate Node test.
- Native Motif interaction regression: 6 passed. Python syntax, notebook schema,
  package asset declarations and whitespace checks passed. Pre-existing saved
  notebook outputs were preserved.

Real browser acceptance used an isolated Jupyter server and scratch notebooks:

- All 46 presets were selected and loaded through the real dropdown, without
  browser errors. Simple and native forms were inspected at 1440, 1100, 850 and
  650 px, across all six steps; no horizontal overflow was detected.
- A measured help hover reached its first painted frame in approximately
  0.6 ms on this browser; this is one local measurement, not a latency guarantee.
  Pinned help did not prevent returning to Input.
- Ten rapid AF2-criterion -> No criteria -> Prepare sequences exported the
  visible empty criterion correctly. A selected project criterion exported its
  definition via --presets. Real browser preset switching cleared a prepared
  compile=true command and disabled Execute while leaving an existing job intact.
- Fetched 1BRS from RCSB, removed waters, kept chains A/D, assigned D as Reference,
  applied D1-D3 with Contig 5/D1-3/7 and D1 CA-only coordinates, computed 26 A-chain
  contacts at 6 A, then checked and saved the task. This authoring/CPU check does
  not claim that the Motif design was sampled on GPU.
- Raw sample and corresponding successful ESMFold2-fast refold were opened from
  the native result catalogue. Ligand distribution was selected and CPU-checked;
  its selector was absent from a protein/peptide Save panel. Empty-input Mol*
  remained mounted. A fresh kernel recovered completed job logs and safe defaults.

Actual minimum model/runtime acceptance on the 4080 SUPER:

- RCSB 1CRN Context + a P15 design produced two structures using BF16 and
  compile=false. Both relax operations succeeded and result.csv retained the
  original sample1.cif/sample2.cif paths with postprocess_status=succeeded.
- ESMFold2-fast completed Original1 + ordinary MPNN1 for both backbones: four
  complete refolds, no failed predictions. Workers used 0.5 GPU each.
- AF3, AF3-LA and ESMFold2-standard each completed one Original prediction;
  PartiScore completed one actual scoring request. This edited command used
  --limit 1 and 0.5 GPU/worker, one actor per method. AF3 used one recycle,
  MSA off/templates off; standard used one loop and 20 steps. These shortened
  tests prove runtime wiring, not full-default scientific acceptance.
- A separate RCSB 1BRS A Context + P20 design produced one structure. AF2-IG
  model 1, one recycle, completed Original1 + ordinary MPNN1 with no failed
  predictions and 0.5 GPU/worker. The preceding de novo-only complex-mode probe
  was correctly not_applicable; it is not counted as a prediction success.
- USalign actually clustered the two original successful P15 backbones at
  TM threshold 0.5: one pair, two clusters, exit 0. Raw/refold preview and result
  references were checked against real emitted files.
- A modified CPU command was launched through Runtime, then its scratch kernel
  was deleted during its eight-second probe. The tmux process continued and
  exited 0. It also performed model-free criterion refiltering of the saved
  metrics and restored the unfiltered reports. The AF2 project criterion
  evaluated two complete records and passed zero; this is a reporting/threshold
  check, not a benchmark success-rate estimate.

Limits and evidence:

- The existing AF2 chemistry gate rejects 1CRN Context disulfides (e.g. A3 SG--A40
  SG, order 0). No AF2 adapter rule was changed in this UI iteration. This is an
  adapter restriction, not evidence of an intrinsic AF2 inability; the ordinary
  1BRS complex supplied the successful AF2 model run instead.
- Default AF3 online MSA/template search, GPU sampling of all 46 presets,
  compile=true GPU execution, and full-protocol benchmark/scientific performance
  were not claimed or revalidated. No persistent model-cache or dump code was
  changed. Original/FI/ordinary parameter combinations received CPU coverage;
  the real redesign smoke used ordinary MPNN, not every branch/model combination.
- Logs/configs/tests and minimum outputs are under ignored
  .cache/notebook_deep_ux and .cache/notebook_runtime. Browser evidence is kept
  with the test artifacts. Pre-existing user jobs were not stopped or modified.

## Paired and helical benchmark configuration acceptance (2026-10-16)

Five new task files add coarse paired RNA/DNA monomers and NA binders, plus
coarse alpha protein monomers. AME-helical reuses the existing AME alpha file.
The six cohorts retain baseline lengths and counts: 1,000 NA monomers, 600 RNA
bind protein, 600 DNA bind protein, 600 NA bind ligand, 500 protein monomers,
and 4,100 AME designs, for 7,400 configured structures. Each reuses its existing
AF3 validation profile, criteria and Original/FI/ordinary sequence budgets.
The manifest assigns distinct variant output directories. Existing ProtDBench
alpha/beta sampling is retained without resampling.

Actual CPU acceptance:

- Native task expansion produced 62 tasks, 74 distinct length plans and 7,400
  requests. All 74 plans were compiled and collated using one fixed-seed
  representative each: 20 paired representatives and 42 helical representatives
  through six public `partidesign check` calls, plus the remaining 12 monomer
  plans through the same compiler/collator APIs.
- Designed polymer tokens received exactly the intended coarse condition in
  both compiled samples and canonical batches. No token-level secondary
  structure or explicit pairing-partner conditions were introduced. Existing
  AME motif coordinates and fixed identities were retained.
- Parsed task comparisons matched baseline plus only the intended coarse
  condition. Shared AF3 validation files and the existing AME alpha task file
  were unchanged. All six output paths were distinct and absent at acceptance;
  baseline source/provenance records and budgets were preserved.
- Checkpoint loading was forbidden during checks; no model was loaded, and no
  sampling, MSA search, GPU prediction or validation job was launched. No model
  cache, inference or dump code changed. Whitespace checks passed.

These checks cover CPU configuration/feature preparation, not all stochastic
crop/contig outcomes, generated helicity/pairing or scientific success metrics.
Tests, logs and detailed acceptance JSON are in ignored
`.cache/benchmark_condition_variants/paired/` and
`.cache/benchmark_condition_variants/helical/`.
