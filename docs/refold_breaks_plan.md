# Refold 编号与链内断点：代码问题、修改要求和验收计划

日期：2026-10-12。状态：本轮实现和软件验收完成，正式 benchmark 尚未运行。
第 2 节保留修复前审计证据；第 10 节保留原验收计划。实际执行结果和限制见第 12 节，
不能把计划、CPU 编译或软件小样例当作正式 benchmark 科学指标验收。

规范检查仓库：Ubuntu-jzc `/home/megagatlingpea/workdir/partidesign-inference`。
部署检查仓库：jzc-pro5k `/home/zichang/partidesign-inference`。
证据版本为 `HEAD 556110c` 加上一轮已存在的未提交修改，不是纯 HEAD。

这份计划处理结构身份、序列化、拓扑和模型适配的代码正确性。目标是让同一输入的编号、真实断点、条件位置和预测对应关系在完整流水线中一致；不修改模型推理核或训练配置。
范围同时包含 AF2、AF3/AF3-LA 和 ESMFold2-fast/standard。用户追加要求包含项目自定义
`af3_na_bind_ligand` 双 RMSD 标准、criteria 精简、最终 PartiScore 任务级排名及恢复重排名，
以及本轮 benchmark 预算和 AF3 validate YAML 默认项清理。其他已有阈值、比较符、scope、
归并及缺失值行为保持。另按明确请求生成 protein 成员为 prot76 的新 checkpoint 拷贝；
原权重文件和默认 checkpoint 选择保持。

## 本次实现范围

- `source_mapping` 在 preparation 阶段建立源定位、真实图断点和 observed→native 位置映射。
  只有确认的完整序列允许补全；无完整序列的已观测片段使用原生独立 entity，不能按数字差补 X。
- CIF 保留 label 和 author 两套定位。所有实际跨残基聚合物边写入标准 `struct_conn`，
  `struct_conn_type.criteria` 声明该观测边表完整；项目 reader 据此禁止补回缺失边。
  已知完整序列使用 `entity_poly_seq` 和 `pdbx_poly_seq_scheme`，不伪造缺失原子。
- AF2 validation/reward 共用 occurrence ledger 和私有 +50 断点表达。
  有真实断点或 PDB 无法无损表示身份的恢复结果保留 CIF。
- AF3、ESMFold2 保留私有原生完整预测；公共恢复只映射已有源位置，补全位置不加入源评测。
  PAE/PDE、置信度和逻辑设计/target 集合沿同一位置表投影。
- `final_partiscore` 来自实际保留的最终结构总分；普通任务用 `design_groups` 总分，
  抗体用 antigen 总分。全任务最终汇总后才发布 `score_rank` 和 `_score_rankN` 文件名。
  sample/search rank 保留；恢复修改结构后清除整个 task 的旧排名并重新发布。
- dump 热路径不增加文件回读、hash、manifest 或全图重复校验。来源和一致性检查放在
  preparation/验收；列序列化避免重复 CIFColumn 转换。测试和计时均放在忽略的 `.cache/`。

## 1. 已确认的规则

1. **CIF 输入以 label 命名空间为主。** 保留 `label_asym_id` 和源 `label_seq_id`；不能改用 `auth_asym_id` 定位源链，也不能拿内部 token index 重编号已有 context。原生 author 字段可以保留为来源事实，但不能替代 label 主身份。
2. **PDB 输入保留原编号。** 保留 chain ID、`resSeq` 和插入码；不能把作者跳号、插入码或多个相同数字的位置压平。PDB atom serial 如有变化，不影响残基身份；本轮需要保护的是残基/链的定位。
3. **已有 context 的坐标和身份不因编号修复改变。模型采样、flex context 或 relax 原本允许的坐标更新继续遵循既有规则。** 内部 UID、tensor index 可以连续；它们必须通过明确映射关联源定位，不能作为公共文件编号直接写回。
4. **不裁掉 loop 或末端。** 真实缺失片段的断点处理不能变成删除已有残基。DNA→protein 保持已确认的全链 Cα RMSD `<5 Å`，无额外 PAE gate。
5. **AF3 使用原生输入语义。** 不增加官方不接受的 `break_index` JSON 字段，不把单个 `X` 当成断链符号。
6. **保持任务的设计/固定角色。** 源 context 的 gap 处理不允许把 context、固定 FR、motif 或指定原子变成可重新设计的位置。
7. **不增加公共 FASTA、JSON sidecar、hash 或新 manifest。** 必要映射放在现有结构/内存记录或私有 worker 输入中。临时验收文件、prepared input 和运行缓存留在 `.cache/`，不进 Git。

### 1.1 不同格式之间的要求

同格式往返必须保留源定位。跨格式不能以破坏目标格式合法性换取表面编号相同：

- CIF→CIF：源 label 链/残基位置和插入码必须能逐项回溯。
- PDB→PDB：源 chain/resSeq/插入码必须保持，包括合法的零/负编号和编号跳跃。
- PDB→CIF：PDB 源定位必须保留并可逆恢复；必要时在 CIF 合法的 author 字段保存原 PDB resSeq/插入码，同时保持合法、唯一的 label 序列位置。链的公共身份不能被任意更换。
- CIF→PDB：只有在链名、编号、插入码等可无歧义表示时才输出；多字符链名或范围超限不能静默截断、重编号或产生碰撞，应明确报告格式不支持并保留 CIF。

新增设计残基没有源编号，继续使用明确的设计编号规则。已有固定/保留残基和新生成残基不能混用同一来源定位。对于改变长度的局部 redesign，必须显式验证已有定位与新定位的唯一性。

## 2. 审计证据和当前限制

### 2.1 真实源结构的对照案例

| 文件 | 源位置 | C–N 距离（Å） | 正确拓扑 |
| --- | --- | ---: | --- |
| `6sy4_complex.pdb` | A157→A166 | 13.564 | 观测结构存在缺口；两端无直接肽键 |
| 同上 | A175→A180 | 7.458 | 两端无直接肽键 |
| 同上 | B154→B179 | 13.765 | 两端无直接肽键 |
| `4i7y_complex.pdb` | H146→H150 | 8.414 | 两端无直接肽键 |
| 同上 | H217→H219 | 1.327 | 正常肽键；编号跳号不能当作缺口 |

这四个真实缺口和一个跳号反例必须保留为回归样本。不能仅根据数字差值补占位或拆链；缺失序列及其长度应依据完整 polymer sequence 和 observed→full-position 映射确认。

### 2.2 目前已经验证了什么

- reader/compiler 上一轮的保护会在源结构中去掉上述四条错误模板肽键。
- 实际 compiled batch 的四个 `chain_break_after_token` 为 true；H217→H219 为 false，4I7Y 的 45 个带插入码的正常相邻对保持连续。
- 六个 RNA/DNA 设计任务的设计链坐标条件、identity condition 和 batch atom coordinate mask 均未泄漏 native 设计链坐标。
- 实际 `write_cif()`→`read_structure()` CPU 测试确认：输出重新连续编号后，上述四条错误 C–N 键又被模板推断回来。
- 原生 ColabDesign 输入准备确认 AF2 validation 的断点 residue-index 间隔，而非只查看中间 PDB 文件。
- 实际 `prepare_af3_inputs()` 确认 AF3 当前会压缩有缺口的观测序列。

### 2.3 目前没有验证什么

本轮没有加载 AF2/AF3 权重或执行真实 GPU 预测。AF2 输入测试仅将模型构造步骤替换为空操作，保留原生输入准备；它能证明输入 index/坐标/mask，但不能代替完整模型验收，也不证明结构预测准确率。

当前 CPU 检查材料在 `.cache/refold_break_audit/` 和 `.cache/af3_msa_acceptance/`，全部是忽略的临时材料；未来应把必要的验收程序仍放在 `.cache/`，不能添加 tracked fixture/test 目录。

## 3. 问题 P1：源身份被内部连续 index 覆盖

**代码位置：** [reader.py](../src/design/utils/reader.py) `_factor_residue_positions()` / `_rows_to_atom_array()`；[writer.py](../src/design/utils/writer.py) `_author_seq_ids()` / `write_cif()`。

**已确认行为：** reader 对 PDB 生成连续 `res_id`，把原编号另存于 author 注释。writer 又用连续 `res_id` 生成 label/author 编号，已有 context 的源缺口位置无法从输出独立恢复。CIF 原生 label 与 author 命名空间不相同时，也不能任意混用。

**修改方式：**

- 区分源链/残基定位、内部 UID、模型 token index 和后端 residue position。
- 读取时记录实际输入格式及源定位；编译、crop、拼接、decode、score、relax 和 writer 不得丢失该映射。
- writer 对已有 context 按源定位写出；对新增设计位置按设计规则写出。
- 更新依赖公共定位的 hotspot、motif、固定位置、bond locator、结果行和 refold 恢复代码，不能只改 `_atom_site` 的数字。

**硬性要求：** 映射唯一且可逆；插入码不被清空；相同 resSeq 不同插入码不合并；多字符 CIF label 链名不被截断。不得通过改变源坐标、删除残基或修改序列解决定位冲突。

## 4. 问题 P2：拓扑断点无法经过结构文件往返

**代码位置：** reader/compiler 的 `_polymer_template_bonds()`，writer 结构身份和连接输出；[batch.py](../src/design/core/batch.py) `_chain_breaks()`。

**已确认行为：** 四条边读源文件时无键，输出 CIF 再读后却有键：

| 源位置 | 输出归一化位置 | 写出前 | 回读后 | 回读 CA–CA 距离（Å） |
| --- | --- | --- | --- | ---: |
| 6SY4 A157→166 | A155→156 | 无 C–N | 错误 C–N | 14.340 |
| 6SY4 A175→180 | A165→166 | 无 C–N | 错误 C–N | 6.901 |
| 6SY4 B154→179 | B152→153 | 无 C–N | 错误 C–N | 13.804 |
| 4I7Y H146→150 | H146→147 | 无 C–N | 错误 C–N | 9.138 |

**修改方式：**

- 在现有结构身份和标准 polymer 信息中保存足够的源位置/片段事实，使回读能够恢复同一断点。
- 模板连接推断必须服从明确的源断点；不能因为重编号后相邻就补键。
- 明确区分真实显式连接、模板推断连接、source crop 断点和缺失坐标片段。显式合法化学键仍须保留。
- 如果需要添加 CIF 类别，只能使用真实、标准且可被外部 reader 解释的内容；不得伪造原子坐标、假肽键或不被消费的自定义 break 类别。

**硬性要求：** 四条假键在输入、compiled graph、base 输出、回读和 refold 输入各阶段都不能重新出现。H217→H219 正常肽键、插入码连续性、环肽首尾键及合法配体共价连接不能被误删。

几何检测可用于校验或受控兼容旧输出，不能继续作为新输出唯一的断点来源。新生成设计链不能仅因局部几何较差就被擅自拆链。

## 5. 问题 P3：AF2 validation 与 reward 处理不一致

**代码位置：** [validate/af2.py](../src/validate/af2.py) `_private_pdb()` / `_insert_af2_context_break_gaps()`；[af2_runner.py](../src/design/rewards/af2_runner.py) `_write_private_af2_pdb()`；AF2 worker 和 ColabDesign preparation。

**真实 CPU 输入结果：**

| 源位置 | validation 最终 `residue_index` | index 差 | reward index 差 |
| --- | --- | ---: | ---: |
| 6SY4 A157→166 | 155→206 | 51 | 1 |
| 6SY4 A175→180 | 215→266 | 51 | 1 |
| 6SY4 B154→179 | 490→541 | 51 | 1 |
| 4I7Y H146→150 | 146→197 | 51 | 1 |
| 4I7Y 正常 H217→219 | 271→272 | 1 | 1 |

validation 通过私有编号在断点后增加 50；加上正常相邻步长，最终差为 51。CIF 回读虽然重新加了假键，这四例仍被 CA 距离回退识别。reward 没有执行同一处理。

**修改方式：**

- 抽取或复用同一受体断点准备逻辑，由 validation/reward 调用，避免两份独立规则。
- 优先消费已保存的权威断点，保持既有 +50 的 AF2 residue-index 表达；不得把私有 index 写入公共输出定位。
- 同步验证序列、atom37、模板、initial guess 和输出 residue 对应关系。
- 保留 native ColabDesign 多 context preparation 行为；不能把其 target asym 设置凭空认定为新 bug。

**硬性要求：** 四处断点在真实模型输入均为预期 index 间隔，正常相邻处为 1；所有观测残基保留。当前输入核查中 6SY4 为 364 个受体残基，4I7Y 为 277 个，已有 atom37 坐标及 mask 正确对应。cyclic offset 仅影响设计链块，不能覆盖受体断点或链间间隔。

## 6. 问题 P4：AF3 把有缺口的 context 压成连续序列

**代码位置：** [af3_input.py](../src/design/rewards/af3_input.py) `build_af3_input()` / `AF3Input`；[af3_backend.py](../src/design/rewards/af3_backend.py) `prepare_af3_inputs()`。

**已确认行为：** 6SY4 A/B context 长度为 188/176，4I7Y H/L 为 250/27。四处真实缺口两端都变成相邻 AF3 residue position；没有 break、占位或片段拆分。polymer–polymer 键被省略，由 AF3 从 sequence 推断主链。仅修 source BondList 不会解决这个输入问题。当前补一个 `X` 又会因 sequence 长度不等于观测残基数而报错。

### 6.1 必须遵循的 AF3 原生语义

官方 protein JSON 有 chain `id`、`sequence`、MSA、templates 等字段，没有公共 `break_index`/`residue_index` 字段。不同链通过多个独立 sequence entity 提交。

- **真正不相连的输入片段：** 按原生多个 chain entity 表示，并维护这些 native entity 与原始源片段的映射。
- **同一 polymer 中未解析的坐标区域：** 使用确认的完整序列，templates 仅映射已观测位置；不能把缺口两侧直接拼接。
- **身份未知但缺失位置已确认：** 只有在位置/长度得到确认时，才可使用原生未知残基编码；蛋白 `X` 是未知氨基酸，不是断链符号。核酸不能套用蛋白 `X` 的规则。
- **位置/长度或片段关系不明确：** 给出可解释诊断，要求补全输入信息；不得按空间距离猜长度、按作者数字差一律补 X，或静默输出伪连续输入。

不能因为观测结构缺少连接，就自动认定为化学上被切开的独立分子。已知完整序列的未解析 loop 和真实分离片段必须区分。若只提供分离片段而没有完整 polymer 信息，允许按原生链实体表示这些给定片段，但必须明确采用的是片段输入语义，不能声称恢复了原完整链。

**修改方式：** 在 canonical source 记录中保留足够的片段/完整序列映射，在 AF3 preparation 中转换为原生 entity。FM 输入 token/latent ABI 不改变；补全序列、拆分后端 entity 属于 refold 适配，不向 FM 偷加 context token 或坐标。

## 7. 问题 P5：MSA、template 和化学位置仍假设观测序列不变

**代码位置：** `af3_input.py` `msa_queries()` / `attach_msa_references()`；[af3_msa.py](../src/validate/af3_msa.py)；AF3 native template/query mapping。

**当前事实：** 缓存是对观测序列的精确命中；A3M query 与请求序列必须相等，template query 索引直接使用。此前 9/9 cache hit 只能证明旧观测 query 的命中，不能保证补全或拆分后的新 query 仍合法命中。

**修改方式：**

- 为每个 native 请求建立明确的 source observed position→native sequence position 映射。
- 补全序列或拆分片段后，同步处理 paired/unpaired MSA 列、template `queryIndices`、修饰位置和配体共价 bond 端点。
- query 完全相同才可直接复用精确缓存。若实施合法的 MSA/template 投影，必须验证 A3M query、列长和索引；不得修改共享原缓存伪造 hit。
- 必要的新 query 走既有自动搜索和缓存逻辑，不因断点适配清空无关缓存或重建数据库。保持可移植 cache-only/auto 和 templates 开启的既有行为。
- binder/design chain 继续不搜索 MSA；拆 context 片段不能改变设计角色。

**硬性要求：** A3M 行的 query 坐标维度正确，template 索引均在范围内并逐位映射到正确源残基；修饰和 bond endpoint 不因插位而移动到错误残基。缓存状态可被解释，不把错误 query 宣称为命中。

## 8. 问题 P6：预测恢复、PAE 和 RMSD 只支持旧一对一映射

**代码位置：** [af3_worker.py](../src/design/rewards/af3_worker.py) `_restore_authoritative_cif()`；[validate/af3.py](../src/validate/af3.py)；[metrics.py](../src/validate/metrics.py) 及 benchmark metrics。

**已确认限制：** 当前要求源链与 native 链双射，使用 native residue ordinal 访问观测 residue ledger，并要求每个 native 残基都有源原子、覆盖连续。补全或拆链后会报错；简单修改序列或取消 assert 会制造错配。

**修改方式：**

- 显式支持源片段→native entity 和 observed residue→native residue 的映射，必要时是一条源链对应多个后端实体。
- 已观测残基严格恢复到原 label/PDB 定位；补全/占位位置没有源原子，不能伪造源身份或塞进原 reference。
- 输出策略应明确区分后端全预测结构和面向原输入的恢复结构。保持已有 base 样本不变；匹配原 reference 的恢复/评测只使用明确映射的源位置。
- confidence token 排序、PAE 行列和设计/target mask 使用同一映射。多个后端片段仍按原逻辑 target 集合聚合，不能因片段数量不同改变指标权重。
- RMSD 使用真实对应残基/原子；缺失坐标、占位 token 不参与虚假的坐标配对。不能把断点处理变成扩大或缩小现有成功 gate。

**硬性要求：** 无 source/native residue 漂移；已有原子 element/restype 和合法修饰保持一致；公共输出编号、置信度 token 和几何指标使用同一逻辑位置。旧不兼容 prepared/refold 缓存必须被识别，不能静默复用；原始采样不覆盖或删除。

## 9. 必须保住的已有功能

| 相关功能 | 不允许发生的回归 |
| --- | --- |
| 正常连续 context | 无故补 X、拆链、增加 index gap、换序列或改变输入拓扑 |
| 正常作者跳号、插入码 | H217→219 被误断；相同 resSeq 不同插入码被合并 |
| Crop 与 motif scaffolding | crop 片段被错误拼接；motif、hotspot、固定序列/原子定位漂移或泄漏 |
| Antibody VH/VL/VHH | FR/CDR 和 scaffold 定位改变；未指定区域被重新设计；真实多链被拼成一链 |
| Cyclic peptide | head-tail bond/offset 被当成普通断点；AF2/AF3 cyclic offset 波及 context |
| 二硫键和合法共价 ligand | 显式连接被通用 gap 处理误删；bond endpoint 指向错误位置 |
| Protein/RNA/DNA/ligand 输入 | 把不支持的 AF2 modality 伪装成 protein；未知核酸被当蛋白 X；CCD/SMILES身份漂移 |
| Score/relax/validate | 公共定位改变导致原子对齐失败；base 结构覆盖；指标分母或阈值被改动 |
| CLI 与格式约束 | `partidesign` / `python -m src.design` 入口失效；normal 与 antibody 批次限制被取消 |
| 持久 worker 和优化 | 重新加载模型、每样本重新初始化、关闭已有 compile/优化、无必要的磁盘往返 |
| 权重与路由 | checkpoint ABI、EMA、默认路线、solver/seed 行为被顺带修改 |
| 缓存与输出 | 清空全部 MSA/模型 cache；增加公共 FASTA/JSON/hash/fixture目录 |

不支持的输入应明确失败，不能通过静默降级、遗漏残基、猜测序列或忽略断点获得“通过”。

## 10. 验收标准：按阶段执行

以下是未来实施的验收要求，不是已完成的测试清单。每项必须保存可复查的输入、实际最终特征/映射与结果；不能只依赖日志中“成功”字样。

### G0：范围和基线

- 保存修改前受影响模块/配置的可复查基线，记录权重、seed、dtype、device和优化设置；不依赖文件 hash。
- 确认只修改身份/断点/refold 映射范围。之前的 24 处源码/配置/文档修改不能被回滚或混淆为本次新增。
- 选择相同输入进行前后对比；旧科学结果不原地覆盖。

### G1：源身份与格式往返（CPU）

测试至少包含：CIF label/auth 故意不同、多字符 label 链；PDB 跳号、零/负编号、插入码；两个同数字不同插入码残基；连续 context；有 crop 的片段。

通过标准：

- 同格式源 locator 集合完全一致，映射唯一；同一内存结构经过 writer→reader 的 context 坐标误差不超过 `1e-5 Å`。此处比较序列化前后，不要求真实 refold 与原始输入坐标相同。
- 原子名、element、restype 和完整已有显式拓扑按定位比较一致，不用排序后的数组 index 假装身份相同。
- 跨格式合法且可逆，不能表示的 PDB 情况明确失败。
- 用至少一个独立 reader 检查文件合法性和编号，不能只用同一个可能有同样 bug 的项目 reader 验收。

### G2：真实断点往返（CPU）

使用本文件列出的 6SY4/4I7Y 实际输入，经完整读取、任务编译、base writer、回读和 backend preparation。

通过标准：

- 四个真实缺口在各阶段都无跨缺口直接肽键，batch 断点始终正确。
- H217→219 和 45 个已核查插入码正常相邻对不断开，不补 X。
- 显式合法连接仍存在，包括独立小例的长距离显式 CONECT；不能用几何阈值覆盖明确拓扑。
- 无新增源坐标、无删除已有 context；设计链坐标和 identity condition 仍按任务屏蔽。

### G3：AF2 validation/reward 实际 preparation（CPU）

直接核查原生 ColabDesign 的最终 `_inputs`，而不是只看私有 PDB。

通过标准：

- validation/reward 对同一输入给出一致的 context residue index；四处为预期 `Δ51`，正常相邻 `Δ1`。
- 序列与 atom37、template、initial guess 同长度同位置。保留所有观测残基和已有 atom mask。
- 6SY4/4I7Y 的 364/277 receptor 残基基线不减少；各原子坐标与 mask 逐位验证。
- monomer/complex preparation 正常；cyclic offset 仅覆盖设计链块，context index 间隔不变。
- 已有私有 PDB override/输入分支不得绕过权威断点处理；若是明确外部 prepared 输入，其约定和校验必须写清楚。

这仍是输入验收。模型构造替身不能用于 G6 的真实 GPU验收。

### G4：AF3 原生输入与映射（CPU）

通过标准：

- 请求能由当前安装的原生 `folding_input.Input` 解析；无自定义 break JSON 字段。
- 连续链原始输入不变。真实分离片段按独立 native entity 提交；同一 polymer 未解析区域采用确认的完整序列/位置映射。
- 不能一律按 author 编号差补位；不能用一枚 X 表示任意缺口。原生 `X` 是未知残基，`-` 不是合法蛋白 sequence 残基。
- 观测位置、native residue position、template query、修饰位置和 bond endpoint 对应正确。
- 一对多链映射以及没有源原子的补全位置得到显式处理；不再通过取消 assert 容忍错配。
- AF3 与 AF3-LA 共享正确的适配语义；默认权重仍为标准 AF3。

### G5：MSA/template 与 output/metrics（CPU + 已有 cache）

通过标准：

- 所有实际请求 query 与 A3M query 一致，template 索引合法；有明显缺口的例子逐位检查 queryIndices。
- 未变更的 query 继续 auto/cache-only 命中；新 query 正确命中或搜索，不能伪报旧缓存命中。
- 保持 templates=true、搜索并发配置、可移植 cache 和不搜索 binder 的规则。
- 使用原生格式的小型预测文件验证恢复和 confidence 行列重排；这类材料只能作为 CPU 映射测试，不算真实预测。
- mapped context/设计位置覆盖率为 100%；补全位置不被错配到源原子；PAE mask 和 RMSD anchor 集合前后一致。
- 设计链/多段 target 的逻辑角色和 existing criteria/reduction 不因 entity 分段改变。

### G6：真实权重、真实 worker 的小规模 GPU验收

必须加载实际权重，通过公开 CLI或生产 reward 调用进入真实 worker。不得用 fake backend、旧 prediction skip、仅 `--plan` 或 InputOnly preparation 代替。

最小测试组：

| 实例 | 必须运行的路径 | 核查内容 |
| --- | --- | --- |
| 已有连续全蛋白设计结果 | AF2-IG validation + AF2 reward | 真实权重完整完成；正常输入未被错误改写 |
| 真实含 context 断点的全蛋白验收输入 | AF2-IG validation + AF2 reward | 生产 worker 实际 index/mask 与 G3 一致，结构/指标恢复成功 |
| 已有 6SY4 对应设计输出 | 标准 AF3 validation + 对应 reward 输入路径 | 缺口表示、MSA/template、真实预测与源位置恢复正确 |
| 已有 4I7Y 对应设计输出 | 标准 AF3 validation | H/L/D、插入码、正常跳号、真实缺口同时正确 |
| 一条实际 head-tail cyclic peptide | AF2-IG + AF3 | cyclic offset 作用域和输出连接不回归 |
| 普通 protein–ligand 实例 | 标准 AF3；AF3-LA至少一例真实smoke | ligand身份/端点/输出恢复不回归，两个模型共用适配无错 |

AF2 测试必须是其支持的全蛋白设计输入，不能把原 RNA/DNA design benchmark 强行塞进 AF2。若没有可用的真实断点全蛋白输出，应单独准备小规模软件验收任务，不混入正式 benchmark 成功率统计。

通过标准：

- 每项任务完成真实模型推理，得到可读取结构与预期结果行；没有 missing-key、索引越界、序列长度或 restoration 错误。
- 检查最终生产输入和结果映射，不只检查是否进程退出 0。
- 正确性按身份、拓扑、mask、原子覆盖与有限指标验收；不要求这些任意测试任务达到某个生物学 binding success rate。
- 同权重、seed和数值设置的连续对照，输入特征必须一致；可重复的输出/指标变化应在已测定数值重复性容差内。容差须在结果前确定，不能为失败事后放宽。
- 新断点输入预测可能与旧错误输入不同，不用旧错误结果的数值相同作为验收目标。

### G7：现有功能回归和速度（CPU + 代表性真实运行）

- CPU 覆盖 normal protein、peptide、RNA/DNA、ligand、motif、antibody VH/VL/VHH；有 source定位的 hotspot/固定序列/固定原子和显式键都必须回归。
- 对 AF2/AF3 persistent worker 记录初始化次数。同一次 validation/reward 服务生命周期应保持既有热加载复用，不能每样本重新读 checkpoint。
- 不变 shape 的连续输入使用相同 worker、batch和设置测暖机后的至少 20 个请求，分别统计 preparation、预测和整体耗时的中位数及 P95。
- 连续对照 preparation/整体中位数若超过基线 5%，或 P95 超过 10%，应重复一次排除机器负载波动；可复现超限必须定位并处理，不能直接宣称无性能回归。
- 真正补全序列导致 token 数增多的案例与旧错误短序列不直接比较预测速度；报告真实 token 数并做同 shape 对照。不能为了速度再次压平断点。
- 不修改 FM 优化/compile、模型 cache、MPNN worker、路由、seed或GPU资源默认值。修改 schema/索引不应导致全局重编译或重加载模型。

### G8：部署与安装

- 先在 Ubuntu-jzc 完成全部适用检查，再同步到 pro5k。
- 比较部署文件实际内容，不使用 hash 标识代替逻辑验收；确认实际 CLI import 来自目标 checkout。
- 若只改 editable Python源码/文档，通常不需要重装；若变更 dependency/entrypoint/package resources，才更新环境配置并做实际安装验收。必须据最终 diff 判断。
- 运行中的进程不会自动重新导入新源码。先完成当前任务或受控重启相关 worker后再用新代码，不擅自终止其他任务。
- 提交前确认只有计划内 tracked 文件，临时测试和 cache 均 ignored；不引入 `.orig`、tracked tests/fixture 或额外运行产物作为本次修改。

## 11. 实施顺序和完成判据

建议顺序：P1/P2 源身份与断点往返 → P3 统一 AF2 → P4/P5 AF3 原生 preparation 和缓存映射 → P6 恢复/指标 → G6/G7 真实验收 → G8 部署。

前一层有未解决的定位/拓扑错误时，不进入全量 benchmark。可以做独立小型验证，但不能把部分通过描述为全链条完成。

完成必须同时满足：

1. 每项代码问题有对应修改和直接验收证据，普通连续输入不回归。
2. 源文件、base 输出、回读、backend 实际输入和恢复结果使用一致定位/断点。
3. AF2 validation/reward 行为一致；AF3输入、MSA/template和结果映射遵循原生格式。
4. G6 真实权重运行和 G7 热加载检查完成；性能以实际计时和本轮用户确认的 dump/index 重点报告，不能将 preparation 百分比未达标的项描述为全部通过。

## 12. 本轮实际执行结果与限制

证据目录为仓库下被忽略的 `.cache/refold_breaks_fix_20261012_JTZFSu/`。
`goal_acceptance.md` 汇总文件清单、命令、原始日志和未验证部分。

- 源定位、真实断点、完整序列补全、private→source 恢复及 confidence 投影已经实现。
  已实际运行 AF2、原版 AF3、AF3-LA、ESMFold2-fast/standard 权重和生产 worker 小样例，
  覆盖真实 6SY4/4I7Y 输入、连续/环肽、配体及 ESMFold2 完整序列缺口/修饰/混合模态。
- AF3 公共 CLI 另外完成一个隔离小样例，复用真实搜索的 10 aa context 缓存 entry48，
  不搜索 design；开启 templates，实际无 template hit。序列、源身份、主链图和结果行验收通过。
- 最终联合 CPU 回归 1,859 项通过；因后续 AF3 YAML 精简而失效的四项原始 YAML
  字面量断言改为生产解析等价断言后，四项重新通过。该批次排除两项已单独执行的 GPU 门禁。
  criteria、schema 和文件检查不冒充真实模型运行。
- 标准 CIF dump 暖机计时：6SY4 median 73.86→65.26 ms，4I7Y 59.77→52.15 ms，
  小结构 1.86→1.52 ms。ESMFold2 普通小结构恢复发布 CPU median 10.71→2.62 ms。
  真实 20 请求整体对照：AF3 约 +0.2%，ESMFold2-fast +2.4%，AF2 连续 +4.4%、环肽 +2.4%。
  **AF2 tiny preparation 仍为 3.31→4.35 ms，约增加 1.04 ms，未达到原计划 5% 相对门槛。**
  按用户最后确认的重点收束于 index 和 dump；不为该小样本百分比继续扩大模型推理改动。
  AF2/AF3/ESMFold2 对照均复用同一个实际加载模型，原生连续对照输入/输出在既定容差内一致。
- score=true 已实际跑单卡 FM400/PartiScore、OpenMM 成功恢复，以及 pro5k 两卡 NCCL
  FM400/PartiScore 四个跨长度样本。两 rank 各只加载一次 scorer、各处理两个文件；
  最终 CSV/CIF/interface 对应及 validate 回读通过。search/tie/antibody/relax 失败和
  表发布故障路径有 CPU 控制测试；未将这些测试称为对应真实 GPU 场景。
- 本轮任务生产展开为完整三版本 39,620、normal 19,540；2,797 个任务与长度组合均实际
  CPU 编译了首个请求，未遍历所有随机 sample slot。12 份 AF3 validate 删除 168 处
  默认项，28 项生产解析等价测试通过；内部 1800/14400 秒超时仍存在。
- 已按逐文件实际内容比较同步本轮拥有的源码/配置/文档到 pro5k，保留无关改动，未重装
  环境或停止用户进程。Ubuntu 的 `ckpts/partidesign-new.ckpt` 仅替换 prot73→prot76；
  新 prot 458 个张量、共享 AE 829 个张量及其他五成员均逐张量核对，六 route 严格 CPU 加载通过。

尚未执行本轮正式采样、MPNN/refold 或科学成功率验收。GPU 软件门禁使用明确缩小的
recycles/ESMFold2 loops/steps，因此不代表默认精度或科学指标；新 prot76 综合包尚未 GPU
推理。新 checkpoint 未改变默认选择，历史输出未追溯改名。
5. 用户已确认的 loops、criteria、设计角色、公共输出及目录要求保持。
6. Ubuntu 与 pro5k 的实际部署代码一致；未解决项明确列出，不静默跳过或降级。

## 12. 官方依据

- [AF3 输入格式](https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md)：每条 protein entity 的 id/sequence、MSA和模板字段；template queryIndices/templateIndices；没有公共 chain-break/residue-index 字段。
- [AF3 folding_input](https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/common/folding_input.py)：protein JSON schema 与未知残基 X。
- [AF3 structure parsing](https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/structure/parsing.py) 与 [features](https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/model/features.py)：由序列构造原生 residue/token 位置。

正式实施时应核对仓库随带 AF3代码版本，不能因为在线 main 更新而无意改变安装版本或现有模型接口。本计划不要求升级模型包。
