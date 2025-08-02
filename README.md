# 增强版EEG轨迹跟踪算法对比系统

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-enhanced-brightgreen.svg)

## 🎯 项目概述

这是一个先进的EEG脑电地形图运动轨迹分析系统，具有全面的算法对比功能。该增强版本提供了多种跟踪算法的系统性评估，包含改进的可视化和详细的性能分析。

### 核心特性

- **5种先进跟踪算法**：贪婪算法、匈牙利算法、卡尔曼算法、重叠算法和混合算法
- **全面性能分析**：详细指标和统计对比
- **增强可视化**：专业级图表和交互式分析
- **性能优化**：可配置帧数限制和内存管理
- **国际化界面**：全英文标签，确保系统兼容性
- **详细文档**：完整的使用指南和API文档

## 🚀 快速开始

### 1. 安装配置

```bash
# 克隆项目
git clone <repository-url>
cd eeg-trajectory-analysis

# 安装依赖
pip install -r requirements.txt

# 系统验证
python quick_test.py
```

### 2. 快速演示

```bash
# 快速测试模式（推荐首次运行）
python main.py --fast-mode

# 查看结果
ls -la results/
```

### 3. 完整分析

```bash
# 完整算法对比
python main.py --subjects 6 --algorithms greedy hungarian kalman

# 自定义配置
python main.py --frames 150 --epochs 2 --subjects 3
```

## 📊 算法对比功能

### 支持的算法

| 算法 | 描述 | 最适用场景 |
|------|------|------------|
| **贪婪算法** | 快速局部优化 | 实时处理 |
| **匈牙利算法** | 全局最优匹配 | 高精度任务 |
| **卡尔曼算法** | 基于运动预测 | 可预测轨迹 |
| **重叠算法** | 区域重叠分析 | 形状稳定目标 |
| **混合算法** | 多特征融合 | 复杂场景 |

### 性能指标

- **轨迹数量**：检测到的轨迹数量
- **质量分数**：平均轨迹质量 (0-1)
- **处理时间**：计算效率
- **内存使用**：资源消耗
- **稳定性**：运行间一致性

## 🛠️ 配置说明

### 基本配置

```python
# config.py - 关键设置
MAX_SUBJECTS = 6              # 处理的被试数量
MAX_FRAMES_PER_EPOCH = 200    # 每个epoch的帧数限制
COMPARISON_ALGORITHMS = [      # 要对比的算法
    'greedy', 'hungarian', 'kalman', 'overlap', 'hybrid'
]
```

### 命令行选项

```bash
# 基本用法
python main.py [选项]

# 选项说明：
--subjects N        # 最大被试数量
--epochs N          # 每个被试的最大epoch数
--frames N          # 每个epoch的最大帧数
--algorithms LIST   # 指定要测试的算法
--fast-mode         # 快速测试模式
--disable-comparison # 单算法模式
```

## 📁 项目结构

```
eeg-trajectory-analysis/
├── main.py                    # 增强版主程序
├── config.py                  # 配置设置
├── quick_test.py              # 系统验证
├── algorithm_comparison.py    # 增强对比模块
├── requirements.txt           # 依赖库
├── src/                       # 核心模块
│   ├── data_loader.py        # EEG数据加载
│   ├── topography.py         # 地形图生成
│   ├── trajectory_analysis.py # 分析工具
│   └── visualization.py      # 增强绘图
├── trackers/                  # 算法实现
│   ├── base_tracker.py       # 基础跟踪器
│   ├── greedy_tracker.py     # 贪婪算法
│   ├── hungarian_tracker.py  # 匈牙利算法
│   ├── kalman_tracker.py     # 卡尔曼算法
│   ├── overlap_tracker.py    # 重叠算法
│   ├── hybrid_tracker.py     # 混合算法
│   └── tracker_factory.py    # 工厂模式
├── results/                   # 输出目录
│   ├── trajectories/         # 轨迹图
│   ├── algorithm_comparison/  # 对比结果
│   ├── topographies/         # 地形图
│   └── analysis/             # 统计分析
└── logs/                     # 系统日志
```

## 📈 输出和结果

### 生成的文件

1. **轨迹可视化**：各算法的轨迹图
2. **对比图表**：并排算法性能对比
3. **统计分析**：详细性能指标
4. **雷达图**：多维算法比较
5. **CSV数据**：原始指标供进一步分析
6. **综合报告**：文本格式分析总结

### 主要输出文件

```
results/
├── algorithm_comparison/
│   ├── enhanced_comparison_report.txt    # 详细分析报告
│   ├── main_algorithm_comparison.png     # 性能对比图
│   ├── performance_radar_chart.png       # 雷达对比图
│   ├── detailed_comparison_table.png     # 总结表格
│   ├── statistical_analysis.png          # 统计图
│   └── algorithm_metrics.csv             # 原始数据
├── trajectories/                         # 个别结果
└── experiment_summary.png                # 总体摘要
```

## 🎨 可视化示例

### 算法性能对比
- 多面板性能图表
- 统计分布分析
- 效率和质量权衡
- 处理时间比较

### 轨迹分析
- 各算法轨迹图
- 叠加对比可视化
- 质量分数分布
- 运动模式分析

## 🔧 高级用法

### 自定义算法配置

```python
# 在config.py中修改算法参数
ALGORITHM_CONFIGS = {
    'greedy': {
        'distance_threshold': 25.0,
        'enable_reconnection': True,
    },
    'hungarian': {
        'distance_threshold': 25.0,
        'use_weighted_cost': True,
    }
    # ... 其他算法
}
```

### 性能优化

```python
# 内存优化
MAX_FRAMES_PER_EPOCH = 150     # 减少以加快处理
MAX_SUBJECTS = 3               # 限制被试数量进行测试

# 启用优化
ENABLE_ALGORITHM_COMPARISON = True
VISUALIZATION_CONFIG = {
    'generate_comparison_plots': True,
    'create_summary_animations': False  # 禁用以提高速度
}
```

## 🧪 测试和验证

### 系统验证

```bash
# 综合系统测试
python quick_test.py

# 预期输出：
# ✅ 所有依赖正确安装！
# ✅ 5/5算法可用
# ✅ 合成数据处理成功
# ✅ 增强可视化功能正常
```

### 性能基准测试

系统包含内置性能基准测试，用于评估：
- 算法执行速度
- 内存效率
- 结果质量
- 系统稳定性

## 📊 数据要求

### 支持的格式

- **MNE兼容格式**：.fif, .edf, .vhdr, .set
- **标准布局**：10-20, 10-10, 自定义电极位置
- **数据结构**：推荐BIDS兼容组织

### 最低要求

- **通道数**：16+个EEG电极
- **采样率**：250+ Hz
- **持续时间**：每个epoch 1+秒
- **格式**：预处理、无伪迹数据

## 🏃‍♂️ 性能建议

### 快速结果
```bash
python main.py --fast-mode --subjects 2
```

### 高质量
```bash
python main.py --frames 300 --epochs 3 --algorithms hungarian hybrid
```

### 内存限制系统
```bash
python main.py --frames 100 --subjects 2 --epochs 1
```

## 🐛 故障排除

### 常见问题

1. **内存错误**：减少 `MAX_FRAMES_PER_EPOCH` 和 `MAX_SUBJECTS`
2. **未检测到轨迹**：检查数据质量和电极位置
3. **可视化问题**：确保matplotlib后端兼容性
4. **字体问题**：系统使用英文标签以确保兼容性

### 调试模式

```bash
# 启用详细日志
python main.py --subjects 1 --epochs 1 2>&1 | tee debug.log
```

### 系统要求

- **Python**：3.8+（推荐3.9+）
- **内存**：4+ GB RAM
- **存储**：1+ GB可用空间
- **操作系统**：Windows, macOS, Linux

## 📚 文档说明

### API文档
- 每个模块包含全面的文档字符串
- 算法类遵循一致的接口
- 配置选项有完整文档

### 示例笔记本
- 用于交互式分析的Jupyter笔记本
- 逐步算法比较
- 自定义可视化示例

## 🤝 贡献指南

### 开发设置

```bash
# 安装开发依赖
pip install -r requirements.txt
pip install pytest sphinx

# 运行测试
python -m pytest tests/

# 生成文档
sphinx-build -b html docs/ docs/_build/
```

### 代码标准
- 遵循PEP 8
- 全面的文档字符串
- 公共API的类型提示
- 核心功能的单元测试

## 📄 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件。

## 🙏 致谢

- MNE-Python社区提供的EEG处理工具
- OpenCV团队提供的计算机视觉算法
- SciPy/NumPy开发者提供的科学计算工具
- Matplotlib团队提供的可视化能力

## 📞 支持

如有问题、问题或贡献：

1. **文档**：查看此README和内联文档
2. **问题**：为错误和功能请求创建GitHub问题
3. **讨论**：使用GitHub讨论进行问题解答
4. **邮件**：紧急问题联系维护者

---

**版本**：3.1.0 增强版  
**最后更新**：2025年8月  
**状态**：生产就绪