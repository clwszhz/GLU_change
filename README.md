# GLU 版本与 12 个测试矩阵

用于在另一台 GPU 服务器上比较实际耗时和相对初始 GLU 的加速比。

单矩阵 GPU 阶段诊断见 [c120 / ASIC_100ks 跨 GPU 测试规范](docs/profiling/c120-asic100ks/README.md)。目录包含 2026-10-17 的 Mars/hcTracer 对照结果、输入校验清单，以及可直接交给 NVIDIA 端会话的[执行提示词](docs/profiling/c120-asic100ks/PROMPT_NVIDIA.md)。

| 目录 | 内容 |
|---|---|
| `versions/glu-original` | 初始 GLU 基线，包含阶段计时 |
| `versions/c94` | c94 |
| `versions/c104` | c104 |
| `versions/c106` | c106，目标归并更新 |
| `versions/c120` | c120，稀疏前缀与自适应稠密 Schur 后缀 |
| `versions/c140` | c140，稀疏前缀超节点、紧凑操作数流、流水与短归并层融合 |
| `matrices` | 全部 12 个测试矩阵，包含 twotone |
| `references` | 对应参考解，右端项为全 1 |

各版本源码保持原样，未加入新的算法优化。旧编译产物不随包上传。
这些版本此前在 Mars/HPCC 环境中测试；本包没有在 NVIDIA 上进行编译或运行验证。
各版本及第三方依赖的许可证保留在各自目录内。

c120 在原 Mars/HPCC 环境中，旧版 12 个公开矩阵、五套完整冻结复测的数值分解几何平均加速比为 **10.594385 倍**，相对每套重新运行的 GLU。
这不是其他 GPU 的性能保证。机制、逐矩阵结果、原始正式计时样本和验证范围见 [c120 说明](docs/c120.md)及[结果数据](docs/c120-results.json)。

当前 G2 矩阵及参考解已同步为评测器的新版本：阶数 150102、非零元 726674，直接替换旧的 438388 非零元版本。
上面的 c120 历史成绩及 `docs/c120-results.json` 对应旧 G2，不能沿用到当前数据集。矩阵已经准备为完整坐标输入，运行时不再做对称展开。

c140 已在当前新 G2 / 全 12 矩阵、Mars GPU2 / HPCC 环境完成五套固定代码的原完整评测，数值分解几何平均加速比依次为 **13.377625、13.521443、13.183591、13.304726、13.346213 倍**。每矩阵汇总全部 35 个正式样本、先取两侧均值再算 12 矩阵几何平均，结果为 **13.346456 倍**；两种统计口径分别保留。求解正确性、S3 与计时审计均通过。
这些是同一公开 12 例的复测，且与旧 c120 历史成绩的数据和 GPU 不同，不能据此计算两版本的直接收益。开发轮的 **12.680421 倍**也保留。见 [c140 说明](docs/c140.md)及[全部正式样本](docs/c140-results.json)。

## 下载与编译

```bash
git clone https://github.com/clwszhz/GLU_change.git
cd GLU_change
bash build.sh
```

需要 Linux、CUDA Toolkit（提供 `nvcc`）、GCC/G++、make；比较脚本只需 Python 3.7 或更高版本。
默认使用 `-arch=native`；如果当前 CUDA 不支持该选项，明确指定目标架构，例如 A100：

```bash
bash build.sh sm_80
```

也可以只编译两个版本：

```bash
bash build.sh sm_80 glu-original c106
```

只编译初始 GLU 和 c120：

```bash
bash build.sh sm_80 glu-original c120
```

只编译初始 GLU、c120 和 c140：

```bash
bash build.sh sm_80 glu-original c120 c140
```

如果 NVIDIA 编译或运行出现兼容问题，保留错误日志再处理，尤其注意 warp 宽度及同步。

## 测试

先测一个小矩阵：

```bash
CUDA_VISIBLE_DEVICES=0 python3 compare.py --matrices add32_csr --repeats 1
```

再测全部 12 个矩阵（每例 1 次预热、7 次正式）：

```bash
CUDA_VISIBLE_DEVICES=0 python3 compare.py
```

只比较初始 GLU 和 c106：

```bash
CUDA_VISIBLE_DEVICES=0 python3 compare.py --versions glu-original c106
```

比较初始 GLU 和 c120（先小矩阵，再整套）：

```bash
CUDA_VISIBLE_DEVICES=0 python3 compare.py --versions glu-original c120 --matrices add32_csr --repeats 1
CUDA_VISIBLE_DEVICES=0 python3 compare.py --versions glu-original c120
```

比较初始 GLU、c120 和 c140（使用当前新 G2 数据）：

```bash
CUDA_VISIBLE_DEVICES=0 python3 compare.py --versions glu-original c120 c140 --matrices add32_csr --repeats 1
CUDA_VISIBLE_DEVICES=0 python3 compare.py --versions glu-original c120 c140
```

结果保存在终端显示的 `results/日期时间/` 下：

- `times.csv`：每个版本、每个矩阵的数值分解均值、最小/最大值、完整进程耗时及分解加速比，单位 ms。
- `summary.csv`：各版本相对同机初始 GLU 的分解加速比几何平均。
- `samples.jsonl`：逐次时间和解向量相对 L2 检查；每次误差须不超过 1e-6。
- 各次运行的 stdout、stderr、timings.json 和 x.dat。

数值分解时间直接读取 `numeric_factorization`，不乘 0.005；完整进程耗时单独展示。
几何平均基于每个矩阵的“GLU 分解均值 / 候选分解均值”，不把不同矩阵的时间相加。
这是简易比较脚本，未包含原完整评测器的 S2/S3、GPU 利用率和缩放残差检查。

矩阵为原评测使用的一基 COO 文本，第一行为 `行数 列数 非零数`，不是标准 Matrix Market 头。
按原文件直接交给 GLU，保持存储坐标，不做对称展开。`matrices/manifest.json` 记录尺寸与哈希；
`SOURCE_MANIFEST.json` 记录所有复制的源码和数据哈希。
