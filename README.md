# 家庭用电多步时间序列预测

本仓库实现2026年机器学习课程项目：使用过去90天的家庭多变量用电数据，分别直接预测未来90天和365天的日聚合总有功功率。两个预测任务独立训练，不共享模型参数。

实现了三类模型：

1. 两层LSTM基线；
2. Transformer编码器-解码器基线；
3. 自主设计的STL-Former，包括多尺度时间卷积、rFFT频域周期上下文、Transformer全局编码、LSTM时序重读和未来日历条件解码。

## 正式结果

所有结果均为5个随机种子的均值 ± 样本标准差，主要指标只统计原始分钟覆盖率不低于95%的测试日期。本文严格按照课程要求，将分钟级 `global_active_power` 按天求和作为预测目标；因此MAE单位为日聚合 `global_active_power` 求和值，MSE单位为该求和值的平方。若换算为日耗电量，MAE或RMSE可除以60得到kWh/day量级。

| 模型 | 90天 MSE | 90天 MAE | 365天 MSE | 365天 MAE |
|---|---:|---:|---:|---:|
| LSTM | **158948.8 ± 5592.8** | **315.6 ± 5.4** | 137216.8 ± 2117.4 | 287.1 ± 2.5 |
| Transformer | 192321.5 ± 30421.1 | 336.9 ± 36.0 | 136397.2 ± 16910.5 | 278.9 ± 22.7 |
| STL-Former | 177144.4 ± 5723.4 | 322.9 ± 7.5 | **126518.9 ± 2573.2** | **261.9 ± 3.5** |

核心汇总指标和主要图表位于 `results/formal_rerun_all/`，STL-Former消融汇总位于 `results/ablation_stl_former/`。为保持仓库轻量，逐轮训练日志、模型权重和逐种子预测文件未纳入开源版，可通过复现实验重新生成。

## 数据

数据来源：[UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)。仓库包含已聚合的日级数据，便于直接复现实验：

- `data/processed/daily/train_daily.csv`：1075天，2006-12-17至2009-11-25；
- `data/processed/daily/test_daily.csv`：365天，2009-11-26至2010-11-25；
- `data/processed/daily/daily_all.csv`：完整1440天日序列。

原始分钟文件和分钟级派生文件体积较大，不提交到GitHub。需要复现预处理时，将 `household_power_consumption.txt` 放在仓库根目录后运行：

```powershell
python prepare_data.py
```

预处理采用：同日内不超过180分钟的短缺口线性插值；较长缺口使用过去7天同一分钟的中位数；删除首尾两个结构性残缺日；保留每日原始观测覆盖率用于训练和评价掩码。

## 环境

推荐Python 3.11、PyTorch 2.4及CUDA 12.1。可使用：

```powershell
conda env create -f environment.yml
conda activate ihepc-forecast
```

也可以在现有环境中安装 `requirements.txt`。GPU不是必需的，但正式五轮实验建议使用CUDA。

## 文件说明

根目录主要文件如下：

| 文件 | 作用 |
|---|---|
| `README.md` | 项目说明、结果表、运行命令和目录说明 |
| `requirements.txt` | pip依赖列表 |
| `environment.yml` | conda环境配置 |
| `prepare_data.py` | 从原始分钟级文件生成日级数据 |
| `analyze_data.py` | 生成数据概览、相关性和日历模式等分析图 |
| `run_experiments.py` | 主实验入口，运行LSTM、Transformer和STL-Former |
| `run_stl_ablation.py` | STL-Former消融实验入口 |

核心目录如下：

| 目录 | 作用 |
|---|---|
| `src/ihepc_forecast/` | 数据集、训练引擎、绘图工具和模型源码 |
| `data/processed/daily/` | 已处理好的日级训练/测试数据 |
| `figures/` | 报告和README可使用的高清图 |
| `results/formal_rerun_all/` | 正式主实验汇总指标与核心图 |
| `results/ablation_stl_former/` | STL-Former消融实验汇总与图 |

## 复现实验

### 1. 快速验证训练链路

```powershell
python run_experiments.py --smoke
```

烟雾测试只运行一个种子和极少轮数，不能作为报告结果。

### 2. 复现正式主实验

复现三种模型、两种预测长度、五个随机种子的正式主实验：

```powershell
python run_experiments.py `
  --models lstm transformer stl_former `
  --horizons 90 365 `
  --seeds 2026 2036 2046 2056 2066 `
  --epochs 60 `
  --patience 10 `
  --output-dir results/reproduced
```

### 3. 单独运行某个模型或预测长度

```powershell
python run_experiments.py --models lstm --horizons 90
python run_experiments.py --models lstm --horizons 365
python run_experiments.py --models transformer --horizons 90
python run_experiments.py --models transformer --horizons 365
python run_experiments.py --models stl_former --horizons 90
python run_experiments.py --models stl_former --horizons 365
```

### 4. 复现STL-Former消融实验

```powershell
python run_stl_ablation.py `
  --horizons 90 365 `
  --seeds 2026 2036 2046 2056 2066 `
  --epochs 60 `
  --patience 10 `
  --output-dir results/reproduced_ablation
```

### 5. 重新生成日级数据和数据分析图

如果需要从原始分钟级文件重新处理数据，将 `household_power_consumption.txt` 放在仓库根目录，然后运行：

```powershell
python prepare_data.py
python analyze_data.py
```

## 目录

```text
data/processed/daily/     可直接训练的日级数据
figures/                  架构、数据概览、预测曲线和消融图
results/formal_rerun_all/ 正式主实验汇总指标与核心图
results/ablation_stl_former/ STL-Former消融实验汇总与图
src/ihepc_forecast/       数据、训练引擎、绘图及模型源码
analyze_data.py           数据分析图生成入口
prepare_data.py           分钟数据预处理入口
run_experiments.py        主实验入口
run_stl_ablation.py       STL-Former消融实验入口
```

基础模型的实现范式主要参考以下公开资料，并根据本项目的90→90/365任务重新实现：

- [Jason Brownlee多步LSTM教程](https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/)：同一UCI数据集上的LSTM直接向量输出与Encoder-Decoder多步预测；
- [THUML Time-Series-Library](https://github.com/thuml/Time-Series-Library/blob/main/models/Transformer.py)：Vanilla Transformer编码器-解码器长期预测实现；
- [Informer](https://arxiv.org/abs/2012.07436)：已知历史片段、未来零占位符和时间特征组成的生成式解码输入。

相关外部数据、论文和参考实现见上方链接及代码注释中的说明。
