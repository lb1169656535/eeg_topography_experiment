#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG脑电地形图运动轨迹分析主程序 - 算法对比增强版
集成多种跟踪算法对比功能
版本: 3.0.0 - 算法对比版
更新时间: 2025-08-01
"""

import os
import sys
import logging
import json
import pickle
import numpy as np
import gc
import argparse
import platform
from datetime import datetime
from tqdm import tqdm
import warnings
import time

# 抑制警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 添加src到路径
sys.path.append('src')
sys.path.append('trackers')

# 字体配置 - 保持原有设置
def setup_matplotlib_font():
    """配置matplotlib字体"""
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    
    try:
        fm._rebuild()
    except:
        pass
    
    system = platform.system()
    use_chinese = False
    
    chinese_fonts = []
    if system == "Windows":
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
    elif system == "Darwin":  # macOS
        chinese_fonts = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti']
    elif system == "Linux":
        chinese_fonts = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Droid Sans Fallback']
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in chinese_fonts:
        if font in available_fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.text(0.5, 0.5, '测试', ha='center', va='center')
                plt.close(fig)
                use_chinese = True
                print(f"✓ 字体配置成功: {font}")
                break
            except:
                continue
    
    if not use_chinese:
        print("⚠️  使用英文标签模式")
        plt.rcParams['font.family'] = 'DejaVu Sans'
    
    return use_chinese

# 设置字体
USE_CHINESE = setup_matplotlib_font()

from config import Config
from src import EEGDataLoader, TopographyGenerator, TrajectoryAnalyzer, Visualizer
from trackers import TrackerFactory

def get_label(key, chinese_text, english_text):
    """获取标签文本"""
    return chinese_text if USE_CHINESE else english_text

def setup_logging():
    """设置日志系统"""
    log_dir = Config.LOGS_ROOT
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志系统已初始化，日志文件: {log_file}")
    
    return logger

def check_dependencies():
    """检查必要的依赖库"""
    required_packages = {
        'mne': 'MNE-Python',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn',
        'cv2': 'OpenCV',
        'tqdm': 'tqdm'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(name)
    
    if missing_packages:
        print("❌ 缺少以下必要的依赖库:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n请使用以下命令安装:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def print_system_info():
    """打印系统信息"""
    print("=" * 70)
    title = get_label('title', 'EEG脑电地形图运动轨迹分析系统 - 算法对比版', 
                     'EEG Topography Motion Trajectory Analysis System - Algorithm Comparison Edition')
    print(title)
    print("=" * 70)
    print(f"Python版本: {platform.python_version()}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"处理器: {platform.machine()}")
    print(f"字体支持: {'中文' if USE_CHINESE else 'English Only'}")
    
    # 显示实验配置
    summary = Config.get_experiment_summary()
    print(f"\n实验配置:")
    print(f"  被试数量: {summary['total_subjects']}")
    print(f"  对比算法数量: {summary['algorithms_count']}")
    print(f"  算法列表: {', '.join(summary['algorithm_names'])}")
    print(f"  评估指标数量: {summary['metrics_count']}")
    print(f"  每个epoch最大帧数: {summary['max_frames_per_epoch']}")  # 新增显示
    print(f"  算法对比: {'启用' if summary['algorithm_comparison_enabled'] else '禁用'}")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"  总内存: {memory.total / (1024**3):.1f} GB")
        print(f"  可用内存: {memory.available / (1024**3):.1f} GB")
    except ImportError:
        pass
    
    print("=" * 70)

def validate_config():
    """验证配置参数"""
    logger = logging.getLogger(__name__)
    
    # 检查数据目录
    if not os.path.exists(Config.DATA_ROOT):
        error_msg = get_label('data_error', 
                             f"数据目录不存在: {Config.DATA_ROOT}",
                             f"Data directory not found: {Config.DATA_ROOT}")
        logger.error(error_msg)
        print(f"\n❌ {error_msg}")
        print(get_label('check_config', 
                       "请检查config.py中的DATA_ROOT设置",
                       "Please check DATA_ROOT setting in config.py"))
        return False
    
    # 验证算法配置
    validation_results = TrackerFactory.validate_algorithm_config(Config)
    invalid_algorithms = [alg for alg, valid in validation_results.items() if not valid]
    
    if invalid_algorithms:
        logger.warning(f"以下算法配置无效: {invalid_algorithms}")
        print(f"⚠️  以下算法配置可能有问题: {', '.join(invalid_algorithms)}")
    
    # 检查可用算法
    available = TrackerFactory.get_available_algorithms()
    missing = [alg for alg in Config.COMPARISON_ALGORITHMS if alg not in available]
    
    if missing:
        logger.error(f"以下算法不可用: {missing}")
        print(f"❌ 以下算法不可用: {', '.join(missing)}")
        return False
    
    # 验证帧数配置
    if Config.MAX_FRAMES_PER_EPOCH <= 0:
        logger.error(f"无效的最大帧数配置: {Config.MAX_FRAMES_PER_EPOCH}")
        print(f"❌ 无效的最大帧数配置: {Config.MAX_FRAMES_PER_EPOCH}")
        return False
    
    logger.info(f"配置验证完成，最大帧数限制: {Config.MAX_FRAMES_PER_EPOCH}")
    return True

def process_subject_with_multiple_algorithms(data_loader, topo_generator, analyzer, visualizer,
                                           subject_id, sessions, logger):
    """使用多种算法处理单个被试的数据"""
    subject_results = {}
    
    session_label = get_label('session_process', 
                             f"处理被试 {subject_id} (共{len(sessions)}个session, {len(Config.COMPARISON_ALGORITHMS)}种算法)",
                             f"Processing subject {subject_id} ({len(sessions)} sessions, {len(Config.COMPARISON_ALGORITHMS)} algorithms)")
    logger.info(session_label)
    
    # 创建所有跟踪器
    trackers = TrackerFactory.create_all_trackers(Config)
    if not trackers:
        logger.error(f"无法创建跟踪器")
        return None
    
    logger.info(f"成功创建 {len(trackers)} 个跟踪器: {', '.join(trackers.keys())}")
    
    for session_id, session_data in sessions.items():
        session_key = f"{subject_id}_{session_id}"
        session_info = get_label('session_info', 
                                f"  处理session {session_id}",
                                f"  Processing session {session_id}")
        logger.info(session_info)
        
        try:
            epochs = session_data['epochs']
            positions = session_data['positions']
            ch_names = epochs.ch_names
            
            # 选择多个epoch进行分析
            n_epochs_to_analyze = min(len(epochs), Config.MAX_EPOCHS_PER_SUBJECT)
            
            session_algorithm_results = {}
            
            for epoch_idx in range(n_epochs_to_analyze):
                try:
                    epoch_data = epochs.get_data()[epoch_idx]
                    
                    # 生成地形图序列
                    epoch_info = get_label('epoch_topo', 
                                          f"    生成epoch {epoch_idx+1} 地形图序列...",
                                          f"    Generating epoch {epoch_idx+1} topographies...")
                    logger.info(epoch_info)
                    
                    # 使用配置参数限制时间点数量
                    max_time_points = min(epoch_data.shape[1], Config.MAX_FRAMES_PER_EPOCH)
                    epoch_data_subset = epoch_data[:, :max_time_points]
                    
                    logger.info(f"    使用帧数限制: {Config.MAX_FRAMES_PER_EPOCH}, 实际处理: {max_time_points} 帧")
                    
                    topographies = topo_generator.generate_time_series_topographies(
                        epoch_data_subset[np.newaxis, :, :], positions, ch_names
                    )[0]
                    
                    if topographies is None or topographies.size == 0:
                        logger.warning(f"    Epoch {epoch_idx+1}: 地形图生成失败")
                        continue
                    
                    # 标准化地形图
                    for t in range(topographies.shape[0]):
                        topographies[t] = topo_generator.normalize_topography(topographies[t])
                    
                    # 使用每种算法进行轨迹跟踪
                    epoch_algorithm_results = {}
                    
                    for algorithm_name, tracker in trackers.items():
                        try:
                            track_info = get_label('epoch_track',
                                                  f"    使用{algorithm_name}算法跟踪epoch {epoch_idx+1}...",
                                                  f"    Tracking epoch {epoch_idx+1} with {algorithm_name}...")
                            logger.info(track_info)
                            
                            start_time = time.time()
                            tracking_results = tracker.track_sequence(topographies)
                            end_time = time.time()
                            
                            if not tracking_results or 'trajectories' not in tracking_results:
                                logger.warning(f"    {algorithm_name}: Epoch {epoch_idx+1} 轨迹跟踪返回空结果")
                                continue
                            
                            trajectories = tracking_results['trajectories']
                            if not trajectories:
                                logger.warning(f"    {algorithm_name}: Epoch {epoch_idx+1} 未检测到有效轨迹")
                                continue
                            
                            # 记录结果
                            epoch_algorithm_results[algorithm_name] = {
                                'trajectories': trajectories,
                                'metrics': tracking_results.get('metrics', {}),
                                'summary': tracking_results.get('summary', {}),
                                'computation_time': end_time - start_time,
                                'processed_frames': topographies.shape[0]  # 新增：记录实际处理帧数
                            }
                            
                            found_info = get_label('found_traj',
                                                  f"    {algorithm_name}: Epoch {epoch_idx+1} 找到 {len(trajectories)} 条轨迹 "
                                                  f"(处理{topographies.shape[0]}帧, 耗时 {end_time - start_time:.2f}s)",
                                                  f"    {algorithm_name}: Epoch {epoch_idx+1} found {len(trajectories)} trajectories "
                                                  f"(processed {topographies.shape[0]} frames, time: {end_time - start_time:.2f}s)")
                            logger.info(found_info)
                            
                        except Exception as e:
                            logger.error(f"    {algorithm_name}: Epoch {epoch_idx+1} 轨迹跟踪失败: {e}")
                            continue
                    
                    # 如果有结果，保存epoch级别的对比
                    if epoch_algorithm_results:
                        # 保存每种算法的代表性可视化
                        for algorithm_name, results in epoch_algorithm_results.items():
                            trajectories = results['trajectories']
                            
                            # 保存轨迹图
                            traj_path = os.path.join(Config.RESULTS_ROOT, "trajectories", 
                                                   f"{session_key}_epoch{epoch_idx}_{algorithm_name}_trajectories.png")
                            try:
                                title = get_label('traj_title',
                                                f"被试{subject_id} Session{session_id} Epoch{epoch_idx} - {algorithm_name}算法 ({results['processed_frames']}帧)",
                                                f"Subject {subject_id} Session {session_id} Epoch {epoch_idx} - {algorithm_name} Algorithm ({results['processed_frames']} frames)")
                                visualizer.plot_trajectories(
                                    trajectories, topographies.shape[1:],
                                    title=title,
                                    save_path=traj_path
                                )
                            except Exception as e:
                                logger.warning(f"保存{algorithm_name}轨迹图失败: {e}")
                        
                        # 将epoch结果合并到session结果中
                        for algorithm_name, results in epoch_algorithm_results.items():
                            if algorithm_name not in session_algorithm_results:
                                session_algorithm_results[algorithm_name] = {
                                    'trajectories': {},
                                    'total_computation_time': 0,
                                    'epoch_count': 0,
                                    'metrics_sum': {},
                                    'total_frames_processed': 0  # 新增
                                }
                            
                            # 合并轨迹（添加epoch前缀）
                            for traj_id, traj_data in results['trajectories'].items():
                                key = f"epoch{epoch_idx}_{traj_id}"
                                session_algorithm_results[algorithm_name]['trajectories'][key] = traj_data
                            
                            # 累计统计
                            session_algorithm_results[algorithm_name]['total_computation_time'] += results['computation_time']
                            session_algorithm_results[algorithm_name]['epoch_count'] += 1
                            session_algorithm_results[algorithm_name]['total_frames_processed'] += results['processed_frames']
                            
                            # 累计指标
                            for metric, value in results.get('metrics', {}).items():
                                if metric not in session_algorithm_results[algorithm_name]['metrics_sum']:
                                    session_algorithm_results[algorithm_name]['metrics_sum'][metric] = []
                                session_algorithm_results[algorithm_name]['metrics_sum'][metric].append(value)
                    
                    # 内存清理
                    del topographies
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"    Epoch {epoch_idx+1} 处理失败: {e}")
                    continue
            
            # 处理session级别的结果
            if session_algorithm_results:
                # 计算平均指标
                for algorithm_name in session_algorithm_results:
                    alg_result = session_algorithm_results[algorithm_name]
                    
                    # 计算平均指标
                    avg_metrics = {}
                    for metric, values in alg_result['metrics_sum'].items():
                        if values:
                            avg_metrics[metric] = np.mean(values)
                    
                    # 更新结果
                    alg_result['average_metrics'] = avg_metrics
                    alg_result['total_trajectories'] = len(alg_result['trajectories'])
                    alg_result['avg_frames_per_epoch'] = alg_result['total_frames_processed'] / alg_result['epoch_count'] if alg_result['epoch_count'] > 0 else 0
                    
                    session_algorithm_results[algorithm_name] = alg_result
                
                subject_results[session_id] = session_algorithm_results
                
                session_summary = get_label('session_summary',
                                          f"  Session {session_id}: 算法对比完成",
                                          f"  Session {session_id}: Algorithm comparison completed")
                logger.info(session_summary)
                
                # 显示各算法的简要结果
                for algorithm_name, alg_result in session_algorithm_results.items():
                    logger.info(f"    {algorithm_name}: {alg_result['total_trajectories']} 条轨迹, "
                              f"平均耗时 {alg_result['total_computation_time']/alg_result['epoch_count']:.2f}s, "
                              f"平均处理 {alg_result['avg_frames_per_epoch']:.0f} 帧/epoch")
            else:
                logger.warning(f"  Session {session_id}: 所有算法均未找到有效轨迹")
                
        except Exception as e:
            logger.error(f"  处理session {session_id} 时出错: {e}")
            continue
    
    return subject_results if subject_results else None

def create_algorithm_comparison_report(all_results, logger):
    """创建算法对比报告"""
    logger.info("生成算法对比报告...")
    
    try:
        # 收集所有算法的统计数据
        algorithm_stats = {}
        
        for subject_id, sessions in all_results.items():
            for session_id, session_data in sessions.items():
                for algorithm_name, alg_data in session_data.items():
                    if algorithm_name not in algorithm_stats:
                        algorithm_stats[algorithm_name] = {
                            'total_trajectories': [],
                            'computation_times': [],
                            'trajectory_lengths': [],
                            'trajectory_qualities': [],
                            'frames_processed': []  # 新增
                        }
                    
                    # 收集统计数据
                    algorithm_stats[algorithm_name]['total_trajectories'].append(alg_data['total_trajectories'])
                    algorithm_stats[algorithm_name]['computation_times'].append(alg_data['total_computation_time'])
                    algorithm_stats[algorithm_name]['frames_processed'].append(alg_data.get('total_frames_processed', 0))
                    
                    # 收集轨迹统计
                    for traj_data in alg_data['trajectories'].values():
                        algorithm_stats[algorithm_name]['trajectory_lengths'].append(traj_data['length'])
                        algorithm_stats[algorithm_name]['trajectory_qualities'].append(traj_data.get('quality_score', 0))
        
        # 生成报告
        report = []
        report.append("=" * 80)
        report.append("EEG轨迹跟踪算法对比报告")
        report.append("=" * 80)
        report.append(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"对比算法数量: {len(algorithm_stats)}")
        report.append(f"处理被试数量: {len(all_results)}")
        report.append(f"帧数限制设置: {Config.MAX_FRAMES_PER_EPOCH} 帧/epoch")  # 新增
        report.append("")
        
        # 算法性能汇总
        report.append("算法性能汇总:")
        report.append("-" * 50)
        
        performance_ranking = []
        
        for algorithm_name, stats in algorithm_stats.items():
            if not stats['total_trajectories']:
                continue
            
            avg_trajectories = np.mean(stats['total_trajectories'])
            avg_time = np.mean(stats['computation_times'])
            avg_length = np.mean(stats['trajectory_lengths']) if stats['trajectory_lengths'] else 0
            avg_quality = np.mean(stats['trajectory_qualities']) if stats['trajectory_qualities'] else 0
            avg_frames = np.mean(stats['frames_processed']) if stats['frames_processed'] else 0
            
            # 计算综合性能分数
            performance_score = (avg_trajectories * 0.3 + 
                               avg_length * 0.25 + 
                               avg_quality * 0.25 + 
                               (10 / max(avg_time, 0.1)) * 0.2)  # 时间越短分数越高
            
            performance_ranking.append((algorithm_name, performance_score, {
                'avg_trajectories': avg_trajectories,
                'avg_time': avg_time,
                'avg_length': avg_length,
                'avg_quality': avg_quality,
                'avg_frames': avg_frames
            }))
            
            report.append(f"\n{algorithm_name.upper()} 算法:")
            report.append(f"  平均轨迹数量: {avg_trajectories:.2f}")
            report.append(f"  平均计算时间: {avg_time:.3f}s")
            report.append(f"  平均轨迹长度: {avg_length:.1f} 帧")
            report.append(f"  平均轨迹质量: {avg_quality:.3f}")
            report.append(f"  平均处理帧数: {avg_frames:.0f} 帧")  # 新增
            report.append(f"  综合性能分数: {performance_score:.3f}")
        
        # 算法排名
        performance_ranking.sort(key=lambda x: x[1], reverse=True)
        
        report.append("\n算法性能排名:")
        report.append("-" * 30)
        
        for i, (algorithm_name, score, details) in enumerate(performance_ranking, 1):
            report.append(f"{i}. {algorithm_name.upper()}: {score:.3f}")
            if i == 1:
                report.append("   🏆 综合性能最佳")
        
        # 算法特色分析
        report.append("\n算法特色分析:")
        report.append("-" * 30)
        
        if performance_ranking:
            # 最多轨迹
            max_traj_alg = max(performance_ranking, key=lambda x: x[2]['avg_trajectories'])
            report.append(f"检测能力最强: {max_traj_alg[0]} ({max_traj_alg[2]['avg_trajectories']:.1f} 条平均轨迹)")
            
            # 最快速度
            min_time_alg = min(performance_ranking, key=lambda x: x[2]['avg_time'])
            report.append(f"计算速度最快: {min_time_alg[0]} ({min_time_alg[2]['avg_time']:.3f}s 平均时间)")
            
            # 最高质量
            max_quality_alg = max(performance_ranking, key=lambda x: x[2]['avg_quality'])
            report.append(f"轨迹质量最高: {max_quality_alg[0]} ({max_quality_alg[2]['avg_quality']:.3f} 平均质量)")
            
            # 最长轨迹
            max_length_alg = max(performance_ranking, key=lambda x: x[2]['avg_length'])
            report.append(f"跟踪持续最长: {max_length_alg[0]} ({max_length_alg[2]['avg_length']:.1f} 帧平均长度)")
        
        # 使用建议
        report.append("\n使用建议:")
        report.append("-" * 20)
        
        if performance_ranking:
            best_overall = performance_ranking[0][0]
            report.append(f"• 综合推荐: {best_overall} (综合性能最佳)")
            
            # 针对不同需求的推荐
            if len(performance_ranking) > 1:
                fastest = min(performance_ranking, key=lambda x: x[2]['avg_time'])[0]
                highest_quality = max(performance_ranking, key=lambda x: x[2]['avg_quality'])[0]
                most_trajectories = max(performance_ranking, key=lambda x: x[2]['avg_trajectories'])[0]
                
                report.append(f"• 实时处理推荐: {fastest} (速度优先)")
                report.append(f"• 高精度分析推荐: {highest_quality} (质量优先)")
                report.append(f"• 复杂场景推荐: {most_trajectories} (检测能力优先)")
        
        # 参数配置信息
        report.append("\n当前参数配置:")
        report.append("-" * 30)
        report.append(f"• 最大帧数限制: {Config.MAX_FRAMES_PER_EPOCH} 帧/epoch")
        report.append(f"• 最大被试数: {Config.MAX_SUBJECTS}")
        report.append(f"• 最大epoch数: {Config.MAX_EPOCHS_PER_SUBJECT}")
        report.append(f"• 阈值百分位: {Config.THRESHOLD_PERCENTILE}%")
        report.append("")
        
        # 保存报告
        report_text = "\n".join(report)
        report_path = os.path.join(Config.RESULTS_ROOT, "algorithm_comparison", "comparison_report.txt")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"算法对比报告已保存: {report_path}")
        
        # 显示报告预览
        print("\n" + "="*60)
        print("算法对比报告预览:")
        print("="*60)
        preview_lines = report_text.split('\n')[:30]  # 显示前30行
        print('\n'.join(preview_lines))
        if len(report) > 30:
            print("\n... (完整报告请查看文件)")
        print("="*60)
        
        return algorithm_stats, performance_ranking
        
    except Exception as e:
        logger.error(f"生成算法对比报告失败: {e}")
        return {}, []

def create_algorithm_comparison_visualizations(all_results, algorithm_stats, performance_ranking, visualizer, logger):
    """创建算法对比可视化"""
    logger.info("生成算法对比可视化...")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        comparison_dir = os.path.join(Config.RESULTS_ROOT, "algorithm_comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # 1. 算法性能对比柱状图
        if algorithm_stats and performance_ranking:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('算法性能对比分析', fontsize=16, fontweight='bold')
            
            algorithms = [item[0] for item in performance_ranking]
            
            # 轨迹数量对比
            ax = axes[0, 0]
            traj_counts = [np.mean(algorithm_stats[alg]['total_trajectories']) for alg in algorithms]
            bars = ax.bar(algorithms, traj_counts, color='skyblue', alpha=0.7)
            ax.set_title('平均轨迹数量对比')
            ax.set_ylabel('轨迹数量')
            ax.tick_params(axis='x', rotation=45)
            
            for bar, count in zip(bars, traj_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{count:.1f}', ha='center', va='bottom')
            
            # 计算时间对比
            ax = axes[0, 1]
            comp_times = [np.mean(algorithm_stats[alg]['computation_times']) for alg in algorithms]
            bars = ax.bar(algorithms, comp_times, color='lightgreen', alpha=0.7)
            ax.set_title('平均计算时间对比')
            ax.set_ylabel('时间 (秒)')
            ax.tick_params(axis='x', rotation=45)
            
            for bar, time in zip(bars, comp_times):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{time:.2f}s', ha='center', va='bottom')
            
            # 轨迹长度对比
            ax = axes[1, 0]
            traj_lengths = [np.mean(algorithm_stats[alg]['trajectory_lengths']) if algorithm_stats[alg]['trajectory_lengths'] else 0 
                           for alg in algorithms]
            bars = ax.bar(algorithms, traj_lengths, color='orange', alpha=0.7)
            ax.set_title('平均轨迹长度对比')
            ax.set_ylabel('长度 (帧)')
            ax.tick_params(axis='x', rotation=45)
            
            for bar, length in zip(bars, traj_lengths):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{length:.1f}', ha='center', va='bottom')
            
            # 综合性能分数
            ax = axes[1, 1]
            performance_scores = [item[1] for item in performance_ranking]
            bars = ax.bar(algorithms, performance_scores, color='coral', alpha=0.7)
            ax.set_title('综合性能分数')
            ax.set_ylabel('性能分数')
            ax.tick_params(axis='x', rotation=45)
            
            for bar, score in zip(bars, performance_scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{score:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            comparison_path = os.path.join(comparison_dir, "algorithm_performance_comparison.png")
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"性能对比图已保存: {comparison_path}")
            
        # 2. 算法特征雷达图
        if len(performance_ranking) >= 2:
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
            
            # 准备数据
            metrics = ['轨迹数量', '计算速度', '轨迹长度', '轨迹质量']
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            
            for i, (algorithm_name, _, details) in enumerate(performance_ranking[:5]):  # 最多显示5个算法
                values = [
                    details['avg_trajectories'] / max([d[2]['avg_trajectories'] for d in performance_ranking]),  # 标准化
                    (10 / max(details['avg_time'], 0.1)) / max([10 / max(d[2]['avg_time'], 0.1) for d in performance_ranking]),  # 速度越快越好
                    details['avg_length'] / max([d[2]['avg_length'] for d in performance_ranking]),
                    details['avg_quality'] / max([d[2]['avg_quality'] for d in performance_ranking]) if max([d[2]['avg_quality'] for d in performance_ranking]) > 0 else 0
                ]
                
                angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
                values += values[:1]  # 闭合图形
                angles += angles[:1]
                
                ax.plot(angles, values, 'o-', linewidth=2, label=algorithm_name, color=colors[i % len(colors)])
                ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1)
            ax.set_title('算法性能雷达图', size=16, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            
            radar_path = os.path.join(comparison_dir, "algorithm_radar_chart.png")
            plt.savefig(radar_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"算法雷达图已保存: {radar_path}")
        
        logger.info("算法对比可视化完成")
        
    except Exception as e:
        logger.error(f"生成可视化失败: {e}")

def cleanup_memory():
    """清理内存"""
    gc.collect()
    
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        if memory_mb > Config.MEMORY_LIMIT_MB:
            logging.getLogger(__name__).warning(
                f"内存使用量过高: {memory_mb:.1f} MB (限制: {Config.MEMORY_LIMIT_MB} MB)"
            )
            return False
    except ImportError:
        pass
    
    return True

def print_final_summary(all_results, algorithm_stats):
    """打印最终总结"""
    print("\n" + "="*70)
    print("算法对比实验完成总结")
    print("="*70)
    
    # 基本统计
    n_subjects = len(all_results)
    total_sessions = sum(len(sessions) for sessions in all_results.values())
    
    print(f"✓ 成功处理被试数量: {n_subjects}")
    print(f"✓ 总session数量: {total_sessions}")
    print(f"✓ 对比算法数量: {len(algorithm_stats)}")
    print(f"✓ 算法列表: {', '.join(algorithm_stats.keys())}")
    print(f"✓ 帧数限制设置: {Config.MAX_FRAMES_PER_EPOCH} 帧/epoch")
    
    # 显示各算法的总体表现
    print(f"\n各算法总体表现:")
    for algorithm_name, stats in algorithm_stats.items():
        if stats['total_trajectories']:
            avg_trajectories = np.mean(stats['total_trajectories'])
            avg_time = np.mean(stats['computation_times'])
            avg_frames = np.mean(stats['frames_processed']) if stats['frames_processed'] else 0
            print(f"  {algorithm_name}: 平均{avg_trajectories:.1f}条轨迹, 平均耗时{avg_time:.2f}s, 平均处理{avg_frames:.0f}帧")
    
    # 输出路径
    print(f"\n结果保存位置:")
    print(f"  📁 轨迹图对比: {os.path.join(Config.RESULTS_ROOT, 'trajectories')}")
    print(f"  📊 算法对比图: {os.path.join(Config.RESULTS_ROOT, 'algorithm_comparison')}")
    print(f"  📄 对比报告: {os.path.join(Config.RESULTS_ROOT, 'algorithm_comparison', 'comparison_report.txt')}")
    
    print("="*70)
    print("🎉 算法对比实验成功完成！")
    print("="*70)

def main():
    """主实验流程"""
    parser = argparse.ArgumentParser(description='EEG Trajectory Analysis System with Algorithm Comparison')
    parser.add_argument('--subjects', type=int, default=None, 
                       help='Maximum number of subjects to process')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Maximum epochs per subject')
    parser.add_argument('--frames', type=int, default=None,
                       help='Maximum frames per epoch')  # 新增参数
    parser.add_argument('--algorithms', nargs='+', default=None,
                       help='Algorithms to compare', choices=Config.COMPARISON_ALGORITHMS)
    parser.add_argument('--disable-comparison', action='store_true',
                       help='Disable algorithm comparison (use greedy only)')
    
    args = parser.parse_args()
    
    # 打印系统信息
    print_system_info()
    
    # 检查依赖
    if not check_dependencies():
        return 1
    
    # 设置日志
    logger = setup_logging()
    start_msg = get_label('start_experiment',
                         "开始EEG脑电地形图运动轨迹分析实验 - 算法对比版",
                         "Starting EEG topography motion trajectory analysis experiment - Algorithm Comparison Edition")
    logger.info(start_msg)
    
    try:
        # 验证配置
        if not validate_config():
            return 1
        
        # 应用命令行参数
        if args.subjects:
            Config.MAX_SUBJECTS = args.subjects
        if args.epochs:
            Config.MAX_EPOCHS_PER_SUBJECT = args.epochs
        if args.frames:  # 新增：动态设置帧数限制
            Config.set_max_frames(args.frames, 'epoch')
            logger.info(f"帧数限制已设置为: {args.frames}")
        if args.algorithms:
            Config.COMPARISON_ALGORITHMS = args.algorithms
        if args.disable_comparison:
            Config.ENABLE_ALGORITHM_COMPARISON = False
            Config.COMPARISON_ALGORITHMS = ['greedy']
        
        # 初始化组件
        init_msg = get_label('init_components', 
                            "初始化分析组件...",
                            "Initializing analysis components...")
        logger.info(init_msg)
        
        data_loader = EEGDataLoader(Config.DATA_ROOT, Config)
        topo_generator = TopographyGenerator(Config)
        analyzer = TrajectoryAnalyzer(Config)
        visualizer = Visualizer(Config)
        
        # 加载数据
        load_msg = get_label('load_data',
                            "开始加载EEG数据...",
                            "Starting to load EEG data...")
        logger.info(load_msg)
        all_data = data_loader.load_all_subjects(Config.MAX_SUBJECTS)
        
        if not all_data:
            error_msg = get_label('no_data',
                                 "未能加载任何EEG数据，请检查数据路径和格式",
                                 "Failed to load any EEG data, please check data path and format")
            logger.error(error_msg)
            print(f"\n❌ {error_msg}")
            return 1
        
        success_msg = get_label('load_success',
                               f"成功加载 {len(all_data)} 个被试的数据",
                               f"Successfully loaded data from {len(all_data)} subjects")
        logger.info(success_msg)
        
        # 存储所有结果
        all_results = {}
        
        # 处理每个被试
        total_subjects = len(all_data)
        processed_subjects = 0
        
        for subject_id, sessions in tqdm(all_data.items(), desc="Processing subjects"):
            try:
                if Config.ENABLE_ALGORITHM_COMPARISON:
                    subject_results = process_subject_with_multiple_algorithms(
                        data_loader, topo_generator, analyzer, visualizer,
                        subject_id, sessions, logger
                    )
                else:
                    # 使用原有的单算法处理逻辑（这里可以调用原来的process_subject_data函数）
                    logger.info("使用单算法模式（greedy）")
                    # 这里可以添加原有的处理逻辑
                    subject_results = None
                
                if subject_results:
                    all_results[subject_id] = subject_results
                    processed_subjects += 1
                    
                    # 定期清理内存
                    if processed_subjects % 2 == 0:
                        cleanup_memory()
                        progress_msg = get_label('progress',
                                               f"已处理 {processed_subjects}/{total_subjects} 个被试",
                                               f"Processed {processed_subjects}/{total_subjects} subjects")
                        logger.info(progress_msg)
                else:
                    no_result_msg = get_label('no_result',
                                            f"被试 {subject_id} 未产生有效结果",
                                            f"Subject {subject_id} produced no valid results")
                    logger.warning(no_result_msg)
                    
            except Exception as e:
                logger.error(f"处理被试 {subject_id} 时出现严重错误: {e}")
                continue
        
        if processed_subjects == 0:
            no_subjects_msg = get_label('no_subjects',
                                       "没有成功处理任何被试数据",
                                       "No subject data was successfully processed")
            logger.error(no_subjects_msg)
            print(f"\n❌ {no_subjects_msg}")
            return 1
        
        complete_msg = get_label('data_complete',
                                f"数据处理完成，成功处理 {processed_subjects} 个被试",
                                f"Data processing complete, successfully processed {processed_subjects} subjects")
        logger.info(complete_msg)
        
        # 生成算法对比报告和可视化
        if Config.ENABLE_ALGORITHM_COMPARISON and all_results:
            algorithm_stats, performance_ranking = create_algorithm_comparison_report(all_results, logger)
            create_algorithm_comparison_visualizations(all_results, algorithm_stats, performance_ranking, visualizer, logger)
            
            # 打印最终总结
            print_final_summary(all_results, algorithm_stats)
        else:
            logger.info("算法对比已禁用或无有效结果")
        
        # 保存完整结果
        results_path = os.path.join(Config.RESULTS_ROOT, "algorithm_comparison_results.pkl")
        try:
            with open(results_path, 'wb') as f:
                pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"完整结果已保存: {results_path}")
        except Exception as e:
            logger.error(f"保存完整结果失败: {e}")
        
        success_final = get_label('success_final',
                                 "算法对比实验成功完成!",
                                 "Algorithm comparison experiment completed successfully!")
        logger.info(success_final)
        return 0
        
    except KeyboardInterrupt:
        interrupt_msg = get_label('interrupted',
                                 "用户中断了实验",
                                 "Experiment was interrupted by user")
        logger.info(interrupt_msg)
        print(f"\n🛑 {interrupt_msg}")
        return 130
        
    except Exception as e:
        error_final = get_label('unexpected_error',
                               f"实验过程中发生未预期的错误: {e}",
                               f"Unexpected error during experiment: {e}")
        logger.error(error_final)
        print(f"\n❌ {error_final}")
        return 1
        
    finally:
        cleanup_memory()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)