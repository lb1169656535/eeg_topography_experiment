# 增强版EEG轨迹分析 - 详细使用指南

## 🚀 入门指南

### 第一步：系统验证

在开始分析之前，先验证系统安装：

```bash
python quick_test.py
```

预期输出：
```
✅ 所有依赖正确安装！
✅ 5/5算法可用
✅ 合成数据处理成功
✅ 增强可视化功能正常
🎉 极佳！系统完全正常运行！
```

### 第二步：基础分析

用示例数据进行快速测试：

```bash
# 快速模式 - 推荐首次运行
python main.py --fast-mode

# 这将会：
# - 最多处理3个被试
# - 每个被试使用1个epoch
# - 限制为每个epoch 100帧
# - 测试前3种算法
```

### 第三步：自定义分析

根据需要配置分析：

```bash
# 中等规模分析
python main.py --subjects 6 --epochs 2 --frames 200

# 高精度分析
python main.py --algorithms hungarian hybrid --frames 300

# 速度优化分析
python main.py --algorithms greedy --frames 100 --subjects 4
```

## 🎯 算法选择指南

### 根据需求选择

#### 实时应用
```bash
python main.py --algorithms greedy --frames 100
```
- **贪婪算法**：最快处理
- **低内存使用**：适合资源受限环境
- **良好跟踪**：适合大多数应用

#### 高精度研究
```bash
python main.py --algorithms hungarian hybrid --frames 300
```
- **匈牙利算法**：全局最优匹配
- **混合算法**：最佳整体性能
- **更高精度**：更好的轨迹质量

#### 对比研究
```bash
python main.py --algorithms greedy hungarian kalman overlap hybrid
```
- **所有算法**：全面比较
- **统计分析**：详细性能指标
- **发表级别**：专业可视化

## 📊 理解结果

### 结果目录结构

运行分析后，检查`results/`目录：

```
results/
├── algorithm_comparison/           # 主要比较结果
│   ├── enhanced_comparison_report.txt
│   ├── main_algorithm_comparison.png
│   ├── performance_radar_chart.png
│   ├── detailed_comparison_table.png
│   ├── statistical_analysis.png
│   ├── performance_trends.png
│   └── algorithm_metrics.csv
├── trajectories/                   # 个别轨迹图
│   ├── subject1_session1_epoch0_greedy_trajectories.png
│   ├── subject1_session1_epoch0_hungarian_trajectories.png
│   └── ...
├── experiment_summary.png          # 总体实验摘要
└── enhanced_experiment_summary.txt # 详细文本报告
```

### 关键输出文件说明

#### 1. 增强比较报告 (`enhanced_comparison_report.txt`)
```
执行摘要
---------
• 最佳整体性能：HYBRID（分数：0.847）
• 最快算法：GREEDY（0.0234秒平均）
• 最高质量：HUNGARIAN（分数：0.892）
• 最高效率：GREEDY（42.3轨迹/秒）

详细算法分析
-----------
贪婪算法：
性能指标：
  • 平均轨迹检测：4.23 ± 1.12
  • 平均计算时间：0.0234秒 ± 0.0045秒
  • 平均轨迹质量：0.734 ± 0.089
  ...
```

#### 2. 性能比较图表 (`main_algorithm_comparison.png`)
- **6面板可视化**显示：
  - 平均轨迹数量
  - 计算时间比较
  - 质量分数比较
  - 轨迹长度分析
  - 处理效率
  - 整体性能分数

#### 3. 雷达图 (`performance_radar_chart.png`)
- **多维比较**包括：
  - 轨迹检测能力
  - 质量评估
  - 处理速度
  - 效率指标
  - 稳定性测量

#### 4. CSV数据 (`algorithm_metrics.csv`)
供Excel/R/Python进一步分析的原始数值数据：
```csv
Algorithm,Avg_Trajectories,Avg_Computation_Time,Avg_Quality,...
greedy,4.23,0.0234,0.734,...
hungarian,4.56,0.0892,0.847,...
kalman,3.89,0.0456,0.756,...
...
```

## 🔧 配置选项

### 帧数限制配置

控制处理速度和内存使用：

```bash
# 快速处理（适合测试）
python main.py --frames 50

# 平衡处理（推荐）
python main.py --frames 200

# 高细节（较慢但更完整）
python main.py --frames 400
```

**帧数限制的影响：**
- **低值（50-100）**：更快、更少内存、更短轨迹
- **中等值（150-250）**：平衡性能和质量
- **高值（300+）**：较慢、更多内存、更长轨迹

### 被试和Epoch配置

```bash
# 快速测试
python main.py --subjects 2 --epochs 1

# 标准分析
python main.py --subjects 6 --epochs 2

# 综合研究
python main.py --subjects 12 --epochs 3
```

### 算法特定设置

修改`config.py`进行细调：

```python
ALGORITHM_CONFIGS = {
    'greedy': {
        'distance_threshold': 20.0,    # 更严格匹配
        'enable_reconnection': True,   # 允许轨迹重连
    },
    'hungarian': {
        'distance_threshold': 25.0,
        'use_weighted_cost': True,     # 多因素优化
    },
    'kalman': {
        'prediction_weight': 0.5,      # 更强预测
        'distance_threshold': 30.0,
    }
}
```

## 📈 结果解读

### 性能指标解释

#### 轨迹数量
- **越高通常越好**（检测到更多活动）
- **考虑背景**：某些算法可能过度检测
- **范围**：通常每个epoch 2-8条轨迹

#### 质量分数（0-1）
- **0.0-0.3**：质量差，轨迹碎片化
- **0.3-0.6**：质量一般，适合基础分析
- **0.6-0.8**：质量好，适合研究
- **0.8-1.0**：质量极佳，发表级别

#### 计算时间
- **实时要求**：< 0.1秒每epoch
- **交互分析**：< 0.5秒每epoch
- **批处理**：< 2.0秒每epoch可接受

#### 效率（轨迹/秒）
- **高效率**：> 20轨迹/秒
- **中等效率**：10-20轨迹/秒
- **低效率**：< 10轨迹/秒

### 何时选择每种算法

#### 贪婪算法
✅ **适用于：**
- 需要实时处理
- 计算资源有限
- 快速探索性分析
- 简单轨迹模式

❌ **避免用于：**
- 需要高精度
- 复杂重叠轨迹
- 需要发表质量结果

#### 匈牙利算法
✅ **适用于：**
- 需要高精度
- 需要发表质量结果
- 最优匹配重要
- 计算资源可用

❌ **避免用于：**
- 需要实时处理
- 非常大的数据集
- 快速探索性分析

#### 卡尔曼算法
✅ **适用于：**
- 可预测的运动模式
- 通过短暂遮挡的跟踪
- 轨迹连续性重要
- 运动动力学相关

❌ **避免用于：**
- 高度不规则运动
- 静态或慢运动模式
- 无明确运动方向

#### 重叠算法
✅ **适用于：**
- 有区域形状信息
- 需要空间重叠分析
- 基于形状的跟踪重要
- 中等计算预算

❌ **避免用于：**
- 形状信息不可靠
- 需要很快处理
- 基于点的跟踪足够

#### 混合算法
✅ **适用于：**
- 需要最佳结果
- 计算资源可用
- 复杂跟踪场景
- 需要综合分析

❌ **避免用于：**
- 实时约束
- 简单跟踪场景
- 计算资源有限

## 🎨 可视化指南

### 理解图表

#### 轨迹图
- **彩色线条**：不同轨迹
- **圆圈**：起始点
- **方块**：结束点
- **箭头**：运动方向
- **头部轮廓**：电极布局参考

#### 比较图表
- **柱状图**：直接指标比较
- **箱线图**：统计分布
- **雷达图**：多维视图
- **散点图**：权衡分析

### 自定义可视化

修改`config.py`进行自定义可视化：

```python
VISUALIZATION_CONFIG = {
    'generate_comparison_plots': True,
    'generate_heatmaps': True,
    'generate_trajectory_overlays': True,
    'save_individual_results': True,
}
```

## 🐛 常见问题故障排除

### 问题1：未检测到轨迹

**症状：**
```
⚠️ 算法：未检测到轨迹
```

**解决方案：**
1. 检查数据质量和预处理
2. 调整`config.py`中的阈值参数：
   ```python
   THRESHOLD_PERCENTILE = 85  # 降低以获得更敏感的检测
   MIN_REGION_SIZE = 15       # 更小的最小区域大小
   ```
3. 验证电极位置正确
4. 尝试不同算法（某些可能更敏感）

### 问题2：内存错误

**症状：**
```
MemoryError: 无法分配数组
```

**解决方案：**
1. 减少帧数限制：
   ```bash
   python main.py --frames 100
   ```
2. 处理更少被试：
   ```bash
   python main.py --subjects 3
   ```
3. 使用快速模式：
   ```bash
   python main.py --fast-mode
   ```

### 问题3：处理缓慢

**症状：**
- 处理时间很长
- 系统无响应

**解决方案：**
1. 启用测试快速模式
2. 减少计算负载：
   ```bash
   python main.py --algorithms greedy --frames 100
   ```
3. 关闭其他应用以释放内存
4. 检查系统规格

### 问题4：可视化问题

**症状：**
- 空白或损坏的图表
- 字体渲染问题

**解决方案：**
1. 系统使用英文标签以确保兼容性
2. 更新matplotlib：
   ```bash
   pip install --upgrade matplotlib
   ```
3. 检查`results/`目录中的可用磁盘空间

## 📚 高级使用模式

### 模式1：算法开发工作流

```bash
# 1. 快速验证
python quick_test.py

# 2. 快速算法测试
python main.py --fast-mode --algorithms greedy hungarian

# 3. 详细比较
python main.py --subjects 6 --algorithms greedy hungarian kalman

# 4. 发表质量分析
python main.py --frames 300 --epochs 3 --algorithms hungarian hybrid
```

### 模式2：参数优化

```bash
# 测试不同帧数限制
for frames in 100 150 200 250; do
    python main.py --frames $frames --subjects 2 --algorithms greedy
done

# 比较结果找到最优设置
```

### 模式3：批量分析

```bash
# 处理多个数据集
python main.py --subjects 12 --epochs 3 --algorithms hungarian
```

## 💡 最佳实践

### 1. 从小开始
- 总是从新数据集的`--fast-mode`开始
- 在扩展之前验证结果
- 检查样本输出的质量

### 2. 渐进式分析
1. **快速测试**：2个被试，1个epoch，1-2种算法
2. **中等测试**：4-6个被试，2个epoch，3种算法
3. **完整分析**：所有被试，所有epoch，所有算法

### 3. 结果验证
- 检查轨迹图的生物学合理性
- 比较算法结果的一致性
- 在可用时与已知真实情况验证

### 4. 性能监控
- 在处理期间监控内存使用
- 检查处理时间的可扩展性
- 验证结果质量指标

### 5. 文档记录
- 保存使用的配置设置
- 记录任何参数修改
- 保留分析日志以便重现

## 📈 性能建议

### 快速结果配置
```bash
python main.py --fast-mode --subjects 2
```

### 高质量配置
```bash
python main.py --frames 300 --epochs 3 --algorithms hungarian hybrid
```

### 内存限制系统配置
```bash
python main.py --frames 100 --subjects 2 --epochs 1
```

### 平衡配置（推荐）
```bash
python main.py --subjects 12 --epochs 2 --frames 300 --algorithms greedy hungarian kalman hybrid overlap
```

## 📞 获取帮助

### 内置帮助
```bash
python main.py --help
python quick_test.py --help
```

### 日志分析
检查`logs/`目录中的日志文件以获得详细信息：
```bash
tail -f logs/enhanced_experiment_*.log
```

### 系统信息
```bash
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"
```

### 示例运行
```bash
# 运行示例脚本查看系统功能
python example_analysis.py
```

## 💻 硬件建议

### 最低要求
- **CPU**：双核2GHz+
- **RAM**：4GB
- **存储**：1GB可用空间
- **Python**：3.8+

### 推荐配置
- **CPU**：四核3GHz+
- **RAM**：8GB+
- **存储**：2GB+可用空间
- **Python**：3.9+

### 高性能配置
- **CPU**：八核3GHz+
- **RAM**：16GB+
- **存储**：SSD存储
- **Python**：3.10+

---

这个指南应该帮助你充分利用增强版EEG轨迹分析系统。如需更多支持，请参考主README.md或在项目仓库中创建问题。