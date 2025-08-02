# EEG轨迹跟踪算法对比系统 - 详细使用指南

## 📚 目录

1. [快速开始](#快速开始)
2. [数据准备](#数据准备)
3. [配置说明](#配置说明)
4. [运行实验](#运行实验)
5. [结果解读](#结果解读)
6. [高级用法](#高级用法)
7. [故障排除](#故障排除)

## 🚀 快速开始

### 1. 环境检查

运行快速测试脚本验证安装：

```bash
python quick_test.py
```

如果测试通过，您可以继续下面的步骤。

### 2. 最小化测试

```bash
# 使用3个被试快速测试（约5-10分钟）
python main.py --subjects 3 --epochs 2

# 查看结果
ls results/algorithm_comparison/
```

### 3. 完整实验

```bash
# 运行完整的12被试算法对比（约30-60分钟）
python main.py
```

## 📁 数据准备

### 支持的数据格式

系统支持以下EEG数据格式：
- **BrainVision** (`.vhdr`, `.eeg`, `.vmrk`)
- **EDF/EDF+** (`.edf`)
- **MNE-Python FIF** (`.fif`)
- **EEGLAB** (`.set`)
- **Neuroscan CNT** (`.cnt`)

### 数据目录结构

确保您的数据按以下结构组织：

```
data/ds005262/
├── sub-0/
│   ├── ses-0/
│   │   └── eeg/
│   │       ├── sub-0_ses-0_task-innerspeech_eeg.vhdr
│   │       ├── sub-0_ses-0_task-innerspeech_eeg.eeg
│   │       └── sub-0_ses-0_task-innerspeech_eeg.vmrk
│   ├── ses-1/
│   │   └── eeg/
│   └── ...
├── sub-1/
│   └── ...
└── sub-N/
```

### 数据质量要求

- **采样率**: 建议 ≥ 250 Hz
- **通道数**: 建议 ≥ 32 个EEG通道
- **记录长度**: 建议 ≥ 60 秒
- **数据质量**: 预处理过的干净数据效果更好

### 修改数据路径

在 `config.py` 中修改数据路径：

```python
# 修改为您的数据路径
DATA_ROOT = "/path/to/your/eeg/data"
```

## ⚙️ 配置说明

### 基本配置

在 `config.py` 中可以调整以下主要参数：

```python
# 实验规模
MAX_SUBJECTS = 12              # 处理的被试数量
MAX_EPOCHS_PER_SUBJECT = 3     # 每个被试的epoch数量
MAX_SESSIONS_PER_SUBJECT = 5   # 每个被试的session数量

# 算法对比
ENABLE_ALGORITHM_COMPARISON = True
COMPARISON_ALGORITHMS = [
    'greedy',      # 贪婪算法
    'hungarian',   # 匈牙利算法
    'kalman',      # 卡尔曼预测
    'overlap',     # 重叠度匹配
    'hybrid'       # 混合算法
]
```

### 算法参数

每种算法都有独立的参数配置：

```python
ALGORITHM_CONFIGS = {
    'greedy': {
        'distance_threshold': 25.0,      # 距离阈值
        'enable_reconnection': True,     # 启用重连
        'max_inactive_frames': 25        # 最大非活跃帧数
    },
    'hungarian': {
        'distance_threshold': 25.0,      # 距离阈值
        'use_weighted_cost': True,       # 使用加权成本
        'cost_threshold': 50.0           # 成本阈值
    },
    # ... 其他算法配置
}
```

### 性能优化配置

```python
# 内存和性能
MEMORY_LIMIT_MB = 4096         # 内存限制
TOPO_SIZE = (128, 128)         # 地形图尺寸
TIME_WINDOW = 2.0              # 分析时间窗口

# 检测参数
THRESHOLD_PERCENTILE = 88      # 阈值百分位数
MIN_REGION_SIZE = 25           # 最小区域面积
MAX_REGIONS = 6                # 最大跟踪区域数
```

## 🎮 运行实验

### 命令行参数

```bash
# 基本用法
python main.py [选项]

# 可用选项：
--subjects N          # 限制被试数量
--epochs N             # 限制每被试epoch数量
--algorithms ALG1 ALG2 # 指定要对比的算法
--disable-comparison   # 禁用算法对比，仅使用greedy
--output-dir DIR       # 指定输出目录
--help                 # 显示帮助信息
```

### 常用运行模式

#### 1. 快速测试模式
```bash
# 3个被试，2个epoch，约5分钟
python main.py --subjects 3 --epochs 2
```

#### 2. 特定算法对比
```bash
# 只对比贪婪和匈牙利算法
python main.py --algorithms greedy hungarian
```

#### 3. 单算法模式
```bash
# 禁用算法对比，仅使用greedy
python main.py --disable-comparison
```

#### 4. 完整实验
```bash
# 所有被试，所有算法
python main.py
```

### 实时监控

#### 查看进度
```bash
# 在另一个终端查看日志
tail -f logs/experiment_*.log
```

#### 监控资源使用
```bash
# Linux/macOS
top -p $(pgrep -f "python main.py")

# Windows (PowerShell)
Get-Process python | Where-Object {$_.ProcessName -eq "python"}
```

## 📊 结果解读

### 输出目录结构

```
results/
├── algorithm_comparison/          # 算法对比结果
│   ├── comparison_report.txt      # 详细对比报告 ⭐
│   ├── algorithm_performance_comparison.png  # 性能对比图 ⭐
│   ├── algorithm_radar_chart.png  # 雷达图
│   └── algorithm_comparison_results.pkl  # 原始数据
├── trajectories/                  # 轨迹图
│   └── {subject}_{session}_epoch{N}_{algorithm}_trajectories.png
├── topographies/                  # 地形图
└── analysis/                      # 分析报告
```

### 关键结果文件

#### 1. 对比报告 (`comparison_report.txt`)

报告包含以下关键信息：

```
算法性能汇总:
GREEDY 算法:
  平均轨迹数量: 4.20
  平均计算时间: 0.150s
  综合性能分数: 3.45

算法性能排名:
1. HUNGARIAN: 4.12  🏆 综合性能最佳
2. HYBRID: 3.87
3. GREEDY: 3.45

使用建议:
• 综合推荐: hungarian (综合性能最佳)
• 实时处理推荐: greedy (速度优先)
• 高精度分析推荐: hungarian (质量优先)
```

#### 2. 性能对比图

- **柱状图**: 直观对比各项指标
- **雷达图**: 多维度性能展示
- **轨迹图**: 各算法的实际轨迹结果

### 性能指标含义

| 指标 | 含义 | 理想值 |
|------|------|--------|
| trajectory_count | 检测到的轨迹数量 | 适中（太多可能有误检） |
| average_trajectory_length | 平均轨迹长度 | 较长（表示跟踪稳定） |
| tracking_continuity | 跟踪连续性 | 越高越好 |
| trajectory_smoothness | 轨迹平滑度 | 越高越好 |
| computation_time | 计算时间 | 越短越好 |
| trajectory_quality | 综合质量分数 | 越高越好 |

### 算法选择建议

#### 根据应用场景选择：

1. **实时处理场景**
   - 推荐: `greedy`
   - 特点: 速度快，资源消耗低

2. **高精度离线分析**
   - 推荐: `hungarian`
   - 特点: 全局最优，质量高

3. **平滑轨迹追踪**
   - 推荐: `kalman`
   - 特点: 运动预测，轨迹平滑

4. **形状保持跟踪**
   - 推荐: `overlap`
   - 特点: 基于重叠度，形状一致性好

5. **复杂场景综合**
   - 推荐: `hybrid`
   - 特点: 综合多特征，适应性强

## 🔧 高级用法

### 自定义算法

#### 1. 创建新算法类

```python
# trackers/my_tracker.py
from trackers.base_tracker import BaseTracker

class MyTracker(BaseTracker):
    def __init__(self, config):
        super().__init__(config)
        self.algorithm_name = "my_algorithm"
        self.description = "我的自定义算法"
    
    def match_regions(self, current_regions, distance_threshold=None, frame_idx=0):
        # 实现您的匹配逻辑
        matches = []
        # ... 您的算法实现
        return matches
```

#### 2. 注册新算法

```python
# 在main.py中添加
from trackers.my_tracker import MyTracker
TrackerFactory.register_tracker('my_algorithm', MyTracker)

# 在config.py中添加
COMPARISON_ALGORITHMS.append('my_algorithm')
```

### 批量处理脚本

```python
# batch_analysis.py
import subprocess
import os

# 定义不同的实验配置
experiments = [
    {'subjects': 3, 'algorithms': ['greedy', 'hungarian']},
    {'subjects': 6, 'algorithms': ['kalman', 'overlap']},
    {'subjects': 12, 'algorithms': ['hybrid']}
]

for i, exp in enumerate(experiments):
    output_dir = f'results_experiment_{i+1}'
    cmd = [
        'python', 'main.py',
        '--subjects', str(exp['subjects']),
        '--algorithms'] + exp['algorithms'] + [
        '--output-dir', output_dir
    ]
    
    print(f"Running experiment {i+1}: {' '.join(cmd)}")
    subprocess.run(cmd)
```

### 参数优化

```python
# optimize_parameters.py
from config import Config
import itertools

# 定义参数搜索空间
param_space = {
    'THRESHOLD_PERCENTILE': [85, 88, 90, 92],
    'MIN_REGION_SIZE': [20, 25, 30],
    'MAX_REGIONS': [4, 6, 8]
}

# 网格搜索
best_score = 0
best_params = None

for params in itertools.product(*param_space.values()):
    # 设置参数
    Config.THRESHOLD_PERCENTILE = params[0]
    Config.MIN_REGION_SIZE = params[1]
    Config.MAX_REGIONS = params[2]
    
    # 运行实验并评估
    # ... 运行main.py并获取结果
    # score = evaluate_results(results)
    
    # if score > best_score:
    #     best_score = score
    #     best_params = params

print(f"Best parameters: {best_params}, Score: {best_score}")
```

## 🐛 故障排除

### 常见问题及解决方案

#### 1. 内存不足

**症状**: `MemoryError` 或系统变慢
**解决方案**:
```bash
# 减少处理规模
python main.py --subjects 3 --epochs 1

# 或在config.py中调整
MEMORY_LIMIT_MB = 2048
MAX_EPOCHS_PER_SUBJECT = 1
```

#### 2. 数据加载失败

**症状**: "未能加载任何EEG数据"
**解决方案**:
1. 检查数据路径: `ls /path/to/your/data`
2. 检查文件格式是否支持
3. 确认文件权限: `chmod -R 755 /path/to/data`

#### 3. 算法执行失败

**症状**: 某个算法的结果为空
**解决方案**:
```bash
# 查看详细日志
grep "ERROR\|WARNING" logs/experiment_*.log

# 禁用有问题的算法
python main.py --algorithms greedy hungarian  # 只运行稳定的算法
```

#### 4. 中文字体显示问题

**症状**: 图表中中文显示为方块
**解决方案**:
- 系统会自动检测并回退到英文
- 手动安装中文字体或忽略此问题

#### 5. 依赖库版本冲突

**症状**: ImportError 或版本警告
**解决方案**:
```bash
# 创建新的虚拟环境
python -m venv eeg_env
source eeg_env/bin/activate  # Linux/macOS
# 或 eeg_env\Scripts\activate  # Windows

# 重新安装依赖
pip install -r requirements.txt
```

### 性能优化建议

#### 1. 硬件配置

- **内存**: 推荐 8GB+，建议 16GB+
- **CPU**: 多核处理器，建议 4核+
- **存储**: SSD 优于 HDD

#### 2. 软件优化

```python
# 在config.py中调整
# 降低地形图分辨率
TOPO_SIZE = (64, 64)  # 默认 (128, 128)

# 减少处理帧数
TIME_WINDOW = 1.0     # 默认 2.0

# 限制算法数量
COMPARISON_ALGORITHMS = ['greedy', 'hungarian']  # 只选重要算法
```

#### 3. 并行处理

```python
# 在主程序中添加并行处理
from multiprocessing import Pool

def process_subject_parallel(subject_data):
    # 处理单个被试的逻辑
    pass

# 使用进程池
with Pool(processes=4) as pool:
    results = pool.map(process_subject_parallel, subject_list)
```

### 调试技巧

#### 1. 启用详细日志

```python
# 在main.py开头添加
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

#### 2. 内存监控

```python
# 添加内存监控
import psutil
import gc

def check_memory():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / 1024 / 1024:.1f} MB")
    gc.collect()

# 在关键位置调用
check_memory()
```

#### 3. 性能分析

```python
# 使用性能分析器
import cProfile
import pstats

# 运行性能分析
cProfile.run('main()', 'profile_stats')

# 查看结果
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative').print_stats(10)
```

## 📞 获取帮助

如果您遇到问题：

1. **检查日志**: `logs/experiment_*.log`
2. **运行测试**: `python quick_test.py`
3. **查看文档**: 阅读 `README.md`
4. **社区支持**: 在GitHub Issues中提问
5. **联系维护者**: [your.email@example.com]

---

**祝您使用愉快！如果这个系统对您的研究有帮助，欢迎引用相关文献。** 🎉