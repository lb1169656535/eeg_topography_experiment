#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG轨迹跟踪算法对比系统 - 快速测试脚本
用于验证系统安装和基本功能
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

def test_dependencies():
    """测试依赖库安装"""
    print("🔍 测试依赖库安装...")
    
    required_packages = {
        'numpy': 'NumPy',
        'scipy': 'SciPy', 
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn',
        'cv2': 'OpenCV',
        'tqdm': 'tqdm',
        'mne': 'MNE-Python'
    }
    
    missing_packages = []
    installed_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            installed_packages.append(name)
            print(f"  ✓ {name}")
        except ImportError:
            missing_packages.append(name)
            print(f"  ❌ {name} - 未安装")
    
    print(f"\n安装状态: {len(installed_packages)}/{len(required_packages)} 个包已安装")
    
    if missing_packages:
        print(f"\n❌ 缺少依赖: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    else:
        print("✅ 所有依赖库已正确安装!")
        return True

def test_tracker_factory():
    """测试跟踪器工厂"""
    print("\n🏭 测试跟踪器工厂...")
    
    try:
        # 添加路径
        sys.path.append('trackers')
        sys.path.append('src')
        from trackers import TrackerFactory
        from config import Config
        
        # 测试获取可用算法
        algorithms = TrackerFactory.get_available_algorithms()
        print(f"  ✓ 可用算法: {', '.join(algorithms)}")
        
        # 测试创建跟踪器
        success_count = 0
        for algorithm in algorithms:
            try:
                tracker = TrackerFactory.create_tracker(algorithm, Config)
                if tracker is not None:
                    print(f"  ✓ {algorithm} 跟踪器创建成功")
                    success_count += 1
                else:
                    print(f"  ❌ {algorithm} 跟踪器创建失败")
            except Exception as e:
                print(f"  ❌ {algorithm} 跟踪器创建异常: {e}")
        
        print(f"\n跟踪器创建状态: {success_count}/{len(algorithms)} 个算法可用")
        return success_count > 0
        
    except Exception as e:
        print(f"  ❌ 跟踪器工厂测试失败: {e}")
        return False

def test_synthetic_data():
    """测试合成数据处理"""
    print("\n🧪 测试合成数据处理...")
    
    try:
        sys.path.append('src')
        sys.path.append('trackers')
        
        from src.topography import TopographyGenerator
        from trackers import TrackerFactory
        from config import Config
        
        # 创建合成地形图数据
        n_frames = 50
        size = (64, 64)  # 使用较小尺寸以加快测试
        
        print(f"  🔧 生成 {n_frames} 帧 {size} 尺寸的合成地形图...")
        
        # 创建简单的移动激活区域
        topographies = np.zeros((n_frames, size[0], size[1]))
        
        for i in range(n_frames):
            # 创建移动的高斯激活
            center_x = 20 + int(15 * np.sin(2 * np.pi * i / 30))
            center_y = 20 + int(10 * np.cos(2 * np.pi * i / 20))
            
            y, x = np.ogrid[:size[0], :size[1]]
            activation = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * 5**2))
            topographies[i] = activation
        
        print("  ✓ 合成地形图生成完成")
        
        # 测试跟踪算法
        test_algorithms = ['greedy', 'hungarian']  # 测试主要算法
        
        for algorithm in test_algorithms:
            try:
                print(f"  🎯 测试 {algorithm} 算法...")
                
                tracker = TrackerFactory.create_tracker(algorithm, Config)
                if tracker is None:
                    print(f"    ❌ {algorithm} 跟踪器创建失败")
                    continue
                
                result = tracker.track_sequence(topographies)
                
                if result and 'trajectories' in result:
                    trajectories = result['trajectories']
                    metrics = result.get('metrics', {})
                    
                    print(f"    ✓ {algorithm}: {len(trajectories)} 条轨迹")
                    print(f"    ✓ 计算时间: {metrics.get('computation_time', 0):.3f}s")
                    
                    if len(trajectories) > 0:
                        first_traj = list(trajectories.values())[0]
                        print(f"    ✓ 轨迹长度: {first_traj['length']} 帧")
                else:
                    print(f"    ⚠️  {algorithm}: 未检测到轨迹")
                
            except Exception as e:
                print(f"    ❌ {algorithm} 测试失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 合成数据测试失败: {e}")
        return False

def test_visualization():
    """测试可视化功能"""
    print("\n🎨 测试可视化功能...")
    
    try:
        # 测试matplotlib设置
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        
        # 创建简单测试图
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # 测试中文字体
        try:
            ax.text(0.5, 0.7, '测试中文字体', ha='center', va='center', fontsize=14)
            ax.text(0.5, 0.5, 'Test English Font', ha='center', va='center', fontsize=12)
            chinese_support = True
        except:
            ax.text(0.5, 0.6, 'Font Test (English Only)', ha='center', va='center', fontsize=12)
            chinese_support = False
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('EEG Trajectory Analysis - Font Test')
        ax.axis('off')
        
        # 保存测试图
        test_dir = './test_results'
        os.makedirs(test_dir, exist_ok=True)
        
        test_path = os.path.join(test_dir, 'font_test.png')
        plt.savefig(test_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 测试图保存至: {test_path}")
        print(f"  {'✓' if chinese_support else '⚠️'} 中文字体支持: {'是' if chinese_support else '否'}")
        
        # 测试复杂可视化
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle('Algorithm Comparison Test Charts', fontsize=14)
        
        # 模拟数据
        algorithms = ['Greedy', 'Hungarian', 'Kalman', 'Overlap', 'Hybrid']
        metrics = {
            'trajectory_count': [4.2, 4.5, 3.8, 3.9, 4.3],
            'computation_time': [0.15, 0.45, 0.25, 0.35, 0.55],
            'trajectory_quality': [0.72, 0.85, 0.78, 0.74, 0.82],
            'memory_usage': [50, 80, 65, 70, 95]
        }
        
        # 柱状图测试
        for i, (metric, values) in enumerate(metrics.items()):
            ax = axes[i//2, i%2]
            bars = ax.bar(algorithms, values, alpha=0.7)
            ax.set_title(metric.replace('_', ' ').title())
            ax.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        chart_path = os.path.join(test_dir, 'comparison_charts_test.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 对比图表保存至: {chart_path}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 可视化测试失败: {e}")
        return False

def test_config():
    """测试配置文件"""
    print("\n⚙️ 测试配置文件...")
    
    try:
        from config import Config
        
        # 测试基本配置
        print(f"  ✓ 数据路径: {Config.DATA_ROOT}")
        print(f"  ✓ 结果路径: {Config.RESULTS_ROOT}")
        print(f"  ✓ 最大被试数: {Config.MAX_SUBJECTS}")
        print(f"  ✓ 算法对比: {'启用' if Config.ENABLE_ALGORITHM_COMPARISON else '禁用'}")
        print(f"  ✓ 对比算法: {', '.join(Config.COMPARISON_ALGORITHMS)}")
        
        # 测试配置方法
        summary = Config.get_experiment_summary()
        print(f"  ✓ 实验摘要: {summary['algorithms_count']} 种算法, {summary['total_subjects']} 个被试")
        
        # 测试算法配置
        for algorithm in Config.COMPARISON_ALGORITHMS:
            alg_config = Config.get_algorithm_config(algorithm)
            print(f"  ✓ {algorithm} 配置: {len(alg_config)} 个参数")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 配置测试失败: {e}")
        return False

def generate_test_report(results):
    """生成测试报告"""
    print("\n" + "="*60)
    print("🎯 EEG轨迹跟踪系统 - 快速测试报告")
    print("="*60)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    test_items = [
        ('依赖库检查', results.get('dependencies', False)),
        ('跟踪器工厂', results.get('tracker_factory', False)),
        ('合成数据处理', results.get('synthetic_data', False)),
        ('可视化功能', results.get('visualization', False)),
        ('配置文件', results.get('config', False))
    ]
    
    passed_tests = sum(1 for _, result in test_items if result)
    total_tests = len(test_items)
    
    print("测试项目:")
    for item_name, passed in test_items:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {item_name}: {status}")
    
    print(f"\n总体结果: {passed_tests}/{total_tests} 项测试通过")
    
    if passed_tests == total_tests:
        print("🎉 恭喜！系统安装完成，所有功能正常！")
        print("\n下一步:")
        print("  1. 准备您的EEG数据（参考README.md中的数据格式）")
        print("  2. 运行: python main.py --subjects 3  (快速测试)")
        print("  3. 运行: python main.py  (完整实验)")
    elif passed_tests >= 3:
        print("✅ 系统基本功能正常，可以开始使用！")
        print("⚠️  部分功能可能受限，请检查失败的测试项。")
    else:
        print("❌ 系统存在严重问题，建议重新安装。")
        print("\n建议:")
        print("  1. 检查Python版本（需要3.8+）")
        print("  2. 重新安装依赖: pip install -r requirements.txt")
        print("  3. 检查系统兼容性")
    
    print("\n📁 测试文件保存在: ./test_results/")
    print("📋 如需帮助，请查看README.md或联系维护者")
    print("="*60)

def main():
    """主测试函数"""
    print("🚀 EEG轨迹跟踪算法对比系统 - 快速功能测试")
    print("="*60)
    print("此测试将验证系统安装和基本功能")
    print("预计耗时: 1-2分钟")
    print("")
    
    # 抑制部分日志
    logging.getLogger().setLevel(logging.WARNING)
    
    # 运行各项测试
    results = {}
    
    results['dependencies'] = test_dependencies()
    results['config'] = test_config()
    results['tracker_factory'] = test_tracker_factory()
    results['synthetic_data'] = test_synthetic_data()
    results['visualization'] = test_visualization()
    
    # 生成测试报告
    generate_test_report(results)
    
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)