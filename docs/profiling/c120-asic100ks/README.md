# c120 / ASIC_100ks 跨 GPU 阶段诊断

先在 NVIDIA 上测试 c120 的 ASIC_100ks_csr，与 2026-10-17 的 Mars GPU 0 / hcTracer 试测保持输入、计算与统计口径一致。所有耗时单位为原始 ms。

## 下载后的入口

在已有仓库运行 `git pull --ff-only`；首次使用则运行：

```bash
git clone https://github.com/clwszhz/GLU_change.git
cd GLU_change
```

将 [PROMPT_NVIDIA.md](PROMPT_NVIDIA.md) 的内容交给 NVIDIA 端会话执行。所需源码、矩阵和参考解已经在本仓库，无需从原服务器额外下载二进制或矩阵。

| 文件 | 用途 |
|---|---|
| `../../../versions/c120` | 在 NVIDIA 上重新编译的源码 |
| `../../../matrices/ASIC_100ks_csr.mtx` | 固定输入，99190 阶，578890 个非零 |
| `../../../references/ASIC_100ks_csr.dat` | KLU 参考解，右端项为全 1 |
| [protocol.json](protocol.json) | 可移植的仓库相对路径、输入/源码哈希与统计约定 |
| [manifest.json](manifest.json) | 原 Mars 试测清单；绝对路径与二进制哈希仅作溯源 |
| [summary.json](summary.json) | 原 Mars 全部计时样本、阶段归类和内核次数 |
| [stages.csv](stages.csv) | 原 Mars 阶段耗时表 |
| [mars-report.md](mars-report.md) | 原服务器完整试测说明 |

本目录只发布规范与结果摘要。原始 trace、逐次解文件和运行脚本保存在原服务器的 `workspace/case7-evolution/artifacts/c120-hctracer-protocol-20261017-gpu0/`；不随本次提交重复上传。本目录的 `protocol.json` 另记录发布文件的 SHA256。

## 固定执行与计时口径

- 同一份 c120 源码在目标平台编译；保持 FP64、原数值算法和输入读取方式，矩阵不做对称展开，右端项为全 1。
- GPU 空闲时测试，每次启动新进程，沿用缓存，不强制清除 JIT 缓存。CPU 数值库线程数固定为 1。
- 顺序：普通预热、普通1、采集1、采集2、普通2、普通3、采集3；预热不计入均值。
- 普通运行用于报告速度，采集运行用于诊断；分别报告三次均值、样本标准差和采集开销。这是单矩阵诊断，原有全套性能评测流程保持原样。
- 总窗口取 `timings.json` 中唯一的 `numeric_factorization`，末尾包含原有 GPU 完成同步。不用 `Total GPU time` 替代，不乘 0.005。最终 CPU 三角求解、完整因子下载等按原阶段归属。
- 将真实设备执行事件裁剪到该窗口，按功能归类。同阶段和全部设备事件分别取区间并集，报告跨阶段重叠。融合内核单列，不拆造内部时长。
- 总窗口减去设备活动并集，记为“未被设备活动覆盖的区间”，保留未归因状态。
- 每次检查解的相对 L2 误差和缩放残差，均要求不超过 1e-6。保留日志、逐次结果和原始采集报告。

## NVIDIA 采集适配

使用 Nsight Systems，采集 CUDA Graph 内部节点。按本机 `nsys profile --help` 确认支持的选项，可从下列设置开始：

```text
nsys profile --trace=cuda --sample=none --cpuctxsw=none --cuda-graph-trace=node ...
```

该命令的输出路径和被测程序参数由 NVIDIA 端诊断脚本设置，每次使用独立运行目录。节点级采集可能增加运行开销，应在本机实测，不沿用 Mars 的 3.20%。[Nsight Systems 官方说明](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cuda-graph-trace)

Nsight 时间轴需要与原 `numeric_factorization` 窗口可靠对齐，不假设它与 `phase_origin_epoch_ns` 共用原点，也不直接套用 hcTracer 的纳秒解析方式。若确需添加 NVTX 标记，可在独立诊断副本中给已有分解边界加标记，保留原计算顺序和同步；普通速度仍由原候选测量，标记副本及其差异另行记录。无法可靠对齐或缺失节点执行时长时，报告归因不可用。

两平台自适应分支可能因 warp 宽度等差异选择不同路径。记录层数、稠密块大小和实际启用路径；核对两侧符号 L/U 结构及数值配置。输入相同但路径不同的结果应明确标注，不能直接当作相同内核的硬件差距。

## 原 Mars 对照结果

| 项目 | 数值分解均值（ms） | 样本标准差（ms） |
|---|---:|---:|
| 普通运行，3 次 | 45.281140 | 0.088603 |
| 带 hcTracer，3 次 | 46.731649 | 0.047670 |

采集开销约 +3.20%。7 次求解全部通过正确性检查；三份 trace 每份分解窗口含 1565 次内核执行，名称和次数一致，无零时长内核及边界截断，未发现跨阶段重叠。

| 分解内部部分 | 平均耗时（ms） | 占采集分解窗口 |
|---|---:|---:|
| 稀疏源列准备 | 4.692366 | 10.04% |
| 稀疏目标更新 | 35.030495 | 74.96% |
| 源列准备与更新融合内核 | 1.019053 | 2.18% |
| panel 对角块 LU | 1.048603 | 2.24% |
| panel 下方 L21 求解 | 1.565152 | 3.35% |
| panel 右侧 U12 求解 | 0.641287 | 1.37% |
| panel 剩余 A22 更新 | 0.895780 | 1.92% |
| 稀疏／稠密打包与写回 | 0.044971 | 0.10% |
| GPU 清零与状态回传 | 0.053589 | 0.11% |
| 未被设备活动覆盖的区间 | 1.740352 | 3.72% |
| 分解总窗口 | 46.731649 | 100.00% |

NVIDIA 普通运行与 45.281140 ms 比较速度；两侧采集阶段表用于定位瓶颈。若同时比较相对初始 GLU 的加速比，需在同一 NVIDIA GPU 上重测本仓库的 `glu-original`。
