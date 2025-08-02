#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced EEG Topography Motion Trajectory Analysis Main Program
Algorithm Comparison Edition with Improved Visualization and Analysis
Version: 3.1.0 - Enhanced Edition (Fixed)
Updated: 2025-08-02
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

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add paths
sys.path.append('src')
sys.path.append('trackers')

# Font configuration - English only
def setup_matplotlib_font():
    """Configure matplotlib font for English only"""
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    
    try:
        fm._rebuild()
    except:
        pass
    
    # Use safe English fonts only
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    print("✓ Font configuration: English labels only")
    return True

# Set up font
USE_ENGLISH_ONLY = setup_matplotlib_font()

from config import Config
from src import EEGDataLoader, TopographyGenerator, TrajectoryAnalyzer, Visualizer
from trackers import TrackerFactory
from algorithm_comparison import run_enhanced_algorithm_comparison

def setup_logging():
    """Set up logging system"""
    log_dir = Config.LOGS_ROOT
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"enhanced_experiment_{timestamp}.log")
    
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
    logger.info(f"Enhanced logging system initialized: {log_file}")
    
    return logger

def check_dependencies():
    """Check required dependencies"""
    required_packages = {
        'mne': 'MNE-Python',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn',
        'cv2': 'OpenCV',
        'tqdm': 'tqdm',
        'pandas': 'Pandas',
        'seaborn': 'Seaborn'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(name)
    
    if missing_packages:
        print("❌ Missing required dependencies:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install using: pip install -r requirements.txt")
        return False
    
    return True

def print_system_info():
    """Print system information"""
    print("=" * 80)
    print("EEG TOPOGRAPHY MOTION TRAJECTORY ANALYSIS SYSTEM")
    print("Enhanced Algorithm Comparison Edition")
    print("=" * 80)
    print(f"Python Version: {platform.python_version()}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.machine()}")
    print(f"Font Support: English Only (for compatibility)")
    
    # Display experiment configuration
    summary = Config.get_experiment_summary()
    print(f"\nExperiment Configuration:")
    print(f"  • Subjects to Process: {summary['total_subjects']}")
    print(f"  • Algorithm Comparison: {len(summary['algorithm_names'])} algorithms")
    print(f"  • Algorithms: {', '.join(summary['algorithm_names'])}")
    print(f"  • Evaluation Metrics: {summary['metrics_count']}")
    print(f"  • Max Frames per Epoch: {summary['max_frames_per_epoch']}")
    print(f"  • Max Epochs per Subject: {summary['max_epochs_per_subject']}")
    print(f"  • Algorithm Comparison: {'Enabled' if summary['algorithm_comparison_enabled'] else 'Disabled'}")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"  • Total Memory: {memory.total / (1024**3):.1f} GB")
        print(f"  • Available Memory: {memory.available / (1024**3):.1f} GB")
    except ImportError:
        pass
    
    print("=" * 80)

def validate_config():
    """Validate configuration parameters"""
    logger = logging.getLogger(__name__)
    
    # Check data directory
    if not os.path.exists(Config.DATA_ROOT):
        error_msg = f"Data directory not found: {Config.DATA_ROOT}"
        logger.error(error_msg)
        print(f"\n❌ {error_msg}")
        print("Please check DATA_ROOT setting in config.py")
        return False
    
    # Validate algorithm configuration
    validation_results = TrackerFactory.validate_algorithm_config(Config)
    invalid_algorithms = [alg for alg, valid in validation_results.items() if not valid]
    
    if invalid_algorithms:
        logger.warning(f"Invalid algorithm configurations: {invalid_algorithms}")
        print(f"⚠️  Potentially problematic algorithm configurations: {', '.join(invalid_algorithms)}")
    
    # Check available algorithms
    available = TrackerFactory.get_available_algorithms()
    missing = [alg for alg in Config.COMPARISON_ALGORITHMS if alg not in available]
    
    if missing:
        logger.error(f"Unavailable algorithms: {missing}")
        print(f"❌ Unavailable algorithms: {', '.join(missing)}")
        return False
    
    # Validate frame configuration
    if Config.MAX_FRAMES_PER_EPOCH <= 0:
        logger.error(f"Invalid max frames configuration: {Config.MAX_FRAMES_PER_EPOCH}")
        print(f"❌ Invalid max frames configuration: {Config.MAX_FRAMES_PER_EPOCH}")
        return False
    
    logger.info(f"Configuration validation complete, max frames limit: {Config.MAX_FRAMES_PER_EPOCH}")
    return True

def process_subject_with_multiple_algorithms(data_loader, topo_generator, analyzer, visualizer,
                                           subject_id, sessions, logger):
    """Process single subject data with multiple algorithms"""
    subject_results = {}
    
    logger.info(f"Processing subject {subject_id} ({len(sessions)} sessions, {len(Config.COMPARISON_ALGORITHMS)} algorithms)")
    
    # Create all trackers
    trackers = TrackerFactory.create_all_trackers(Config)
    if not trackers:
        logger.error(f"Unable to create trackers")
        return None
    
    logger.info(f"Successfully created {len(trackers)} trackers: {', '.join(trackers.keys())}")
    
    session_progress = 0
    total_sessions = len(sessions)
    
    for session_id, session_data in sessions.items():
        session_progress += 1
        session_key = f"{subject_id}_{session_id}"
        logger.info(f"  Processing session {session_id} ({session_progress}/{total_sessions})")
        
        try:
            epochs = session_data['epochs']
            positions = session_data['positions']
            ch_names = epochs.ch_names
            
            # Select multiple epochs for analysis
            n_epochs_to_analyze = min(len(epochs), Config.MAX_EPOCHS_PER_SUBJECT)
            
            session_algorithm_results = {}
            
            epoch_progress = 0
            for epoch_idx in range(n_epochs_to_analyze):
                epoch_progress += 1
                
                try:
                    epoch_data = epochs.get_data()[epoch_idx]
                    
                    # Generate topography sequence
                    logger.info(f"    Generating epoch {epoch_idx+1} topography sequence...")
                    
                    # Use configuration parameters to limit time points
                    max_time_points = min(epoch_data.shape[1], Config.MAX_FRAMES_PER_EPOCH)
                    epoch_data_subset = epoch_data[:, :max_time_points]
                    
                    logger.info(f"    Using frame limit: {Config.MAX_FRAMES_PER_EPOCH}, processing: {max_time_points} frames")
                    
                    topographies = topo_generator.generate_time_series_topographies(
                        epoch_data_subset[np.newaxis, :, :], positions, ch_names
                    )[0]
                    
                    if topographies is None or topographies.size == 0:
                        logger.warning(f"    Epoch {epoch_idx+1}: topography generation failed")
                        continue
                    
                    # Normalize topographies
                    for t in range(topographies.shape[0]):
                        topographies[t] = topo_generator.normalize_topography(topographies[t])
                    
                    # Use each algorithm for trajectory tracking
                    epoch_algorithm_results = {}
                    algorithm_progress = 0
                    
                    for algorithm_name, tracker in trackers.items():
                        algorithm_progress += 1
                        
                        try:
                            logger.info(f"    Tracking epoch {epoch_idx+1} with {algorithm_name} algorithm "
                                      f"({algorithm_progress}/{len(trackers)})...")
                            
                            start_time = time.time()
                            tracking_results = tracker.track_sequence(topographies)
                            end_time = time.time()
                            
                            if not tracking_results or 'trajectories' not in tracking_results:
                                logger.warning(f"    {algorithm_name}: Epoch {epoch_idx+1} tracking returned empty results")
                                continue
                            
                            trajectories = tracking_results['trajectories']
                            if not trajectories:
                                logger.warning(f"    {algorithm_name}: Epoch {epoch_idx+1} no valid trajectories detected")
                                continue
                            
                            # Record results
                            epoch_algorithm_results[algorithm_name] = {
                                'trajectories': trajectories,
                                'metrics': tracking_results.get('metrics', {}),
                                'summary': tracking_results.get('summary', {}),
                                'computation_time': end_time - start_time,
                                'processed_frames': topographies.shape[0],
                                'tracking_results': tracking_results  # Keep full results for visualization
                            }
                            
                            logger.info(f"    {algorithm_name}: Epoch {epoch_idx+1} found {len(trajectories)} trajectories "
                                      f"(processed {topographies.shape[0]} frames, time: {end_time - start_time:.3f}s)")
                            
                        except Exception as e:
                            logger.error(f"    {algorithm_name}: Epoch {epoch_idx+1} tracking failed: {e}")
                            continue
                    
                    # If results exist, save epoch-level comparisons
                    if epoch_algorithm_results:
                        # Save representative visualizations for each algorithm
                        for algorithm_name, results in epoch_algorithm_results.items():
                            trajectories = results['trajectories']
                            
                            # Save trajectory plot
                            traj_path = os.path.join(Config.RESULTS_ROOT, "trajectories", 
                                                   f"{session_key}_epoch{epoch_idx}_{algorithm_name}_trajectories.png")
                            try:
                                title = f"Subject {subject_id} Session {session_id} Epoch {epoch_idx} - {algorithm_name.upper()} Algorithm ({results['processed_frames']} frames)"
                                visualizer.plot_trajectories(
                                    trajectories, topographies.shape[1:],
                                    title=title,
                                    save_path=traj_path
                                )
                            except Exception as e:
                                logger.warning(f"Failed to save {algorithm_name} trajectory plot: {e}")
                        
                        # Merge epoch results into session results
                        for algorithm_name, results in epoch_algorithm_results.items():
                            if algorithm_name not in session_algorithm_results:
                                session_algorithm_results[algorithm_name] = {
                                    'trajectories': {},
                                    'total_computation_time': 0,
                                    'epoch_count': 0,
                                    'metrics_sum': {},
                                    'total_frames_processed': 0
                                }
                            
                            # Merge trajectories (add epoch prefix)
                            for traj_id, traj_data in results['trajectories'].items():
                                key = f"epoch{epoch_idx}_{traj_id}"
                                session_algorithm_results[algorithm_name]['trajectories'][key] = traj_data
                            
                            # Accumulate statistics
                            session_algorithm_results[algorithm_name]['total_computation_time'] += results['computation_time']
                            session_algorithm_results[algorithm_name]['epoch_count'] += 1
                            session_algorithm_results[algorithm_name]['total_frames_processed'] += results['processed_frames']
                            
                            # Accumulate metrics
                            for metric, value in results.get('metrics', {}).items():
                                if metric not in session_algorithm_results[algorithm_name]['metrics_sum']:
                                    session_algorithm_results[algorithm_name]['metrics_sum'][metric] = []
                                session_algorithm_results[algorithm_name]['metrics_sum'][metric].append(value)
                    
                    # Memory cleanup
                    del topographies
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"    Epoch {epoch_idx+1} processing failed: {e}")
                    continue
            
            # Process session-level results
            if session_algorithm_results:
                # Calculate average metrics
                for algorithm_name in session_algorithm_results:
                    alg_result = session_algorithm_results[algorithm_name]
                    
                    # Calculate average metrics
                    avg_metrics = {}
                    for metric, values in alg_result['metrics_sum'].items():
                        if values:
                            avg_metrics[metric] = np.mean(values)
                    
                    # Update results
                    alg_result['average_metrics'] = avg_metrics
                    alg_result['total_trajectories'] = len(alg_result['trajectories'])
                    alg_result['avg_frames_per_epoch'] = alg_result['total_frames_processed'] / alg_result['epoch_count'] if alg_result['epoch_count'] > 0 else 0
                    
                    session_algorithm_results[algorithm_name] = alg_result
                
                subject_results[session_id] = session_algorithm_results
                
                logger.info(f"  Session {session_id}: algorithm comparison completed")
                
                # Display brief results for each algorithm
                for algorithm_name, alg_result in session_algorithm_results.items():
                    logger.info(f"    {algorithm_name}: {alg_result['total_trajectories']} trajectories, "
                              f"avg time {alg_result['total_computation_time']/alg_result['epoch_count']:.3f}s, "
                              f"avg frames {alg_result['avg_frames_per_epoch']:.0f}/epoch")
            else:
                logger.warning(f"  Session {session_id}: all algorithms failed to find valid trajectories")
                
        except Exception as e:
            logger.error(f"  Error processing session {session_id}: {e}")
            continue
    
    return subject_results if subject_results else None

def create_enhanced_summary_report(all_results, logger):
    """Create enhanced summary report with detailed insights"""
    logger.info("Generating enhanced summary report...")
    
    try:
        # Collect comprehensive statistics
        algorithm_stats = {}
        subject_performance = {}
        
        for subject_id, sessions in all_results.items():
            subject_performance[subject_id] = {}
            
            for session_id, session_data in sessions.items():
                for algorithm_name, alg_data in session_data.items():
                    if algorithm_name not in algorithm_stats:
                        algorithm_stats[algorithm_name] = {
                            'total_trajectories': [],
                            'computation_times': [],
                            'trajectory_lengths': [],
                            'trajectory_qualities': [],
                            'frames_processed': [],
                            'sessions_processed': 0
                        }
                    
                    # Collect statistics
                    algorithm_stats[algorithm_name]['total_trajectories'].append(alg_data['total_trajectories'])
                    algorithm_stats[algorithm_name]['computation_times'].append(alg_data['total_computation_time'])
                    algorithm_stats[algorithm_name]['frames_processed'].append(alg_data.get('total_frames_processed', 0))
                    algorithm_stats[algorithm_name]['sessions_processed'] += 1
                    
                    # Collect trajectory statistics
                    for traj_data in alg_data['trajectories'].values():
                        algorithm_stats[algorithm_name]['trajectory_lengths'].append(traj_data['length'])
                        algorithm_stats[algorithm_name]['trajectory_qualities'].append(traj_data.get('quality_score', 0))
                    
                    # Track subject performance
                    if algorithm_name not in subject_performance[subject_id]:
                        subject_performance[subject_id][algorithm_name] = {
                            'total_trajectories': 0,
                            'total_time': 0,
                            'sessions': 0
                        }
                    
                    subject_performance[subject_id][algorithm_name]['total_trajectories'] += alg_data['total_trajectories']
                    subject_performance[subject_id][algorithm_name]['total_time'] += alg_data['total_computation_time']
                    subject_performance[subject_id][algorithm_name]['sessions'] += 1
        
        # Generate enhanced report
        report = []
        report.append("=" * 100)
        report.append("ENHANCED EEG TRAJECTORY TRACKING ALGORITHM COMPARISON REPORT")
        report.append("=" * 100)
        report.append(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Subjects Processed: {len(all_results)}")
        report.append(f"Total Sessions Analyzed: {sum(len(sessions) for sessions in all_results.values())}")
        report.append(f"Algorithms Compared: {len(algorithm_stats)}")
        report.append(f"Frame Limit Configuration: {Config.MAX_FRAMES_PER_EPOCH} frames/epoch")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("=" * 50)
        
        best_performers = {}
        for category in ['total_trajectories', 'computation_times', 'trajectory_qualities']:
            if category == 'computation_times':
                # Lower is better for computation time
                best_alg = min(algorithm_stats.items(), 
                             key=lambda x: np.mean(x[1][category]) if x[1][category] else float('inf'))
                best_performers[category] = (best_alg[0], np.mean(best_alg[1][category]))
            else:
                # Higher is better for other metrics
                best_alg = max(algorithm_stats.items(), 
                             key=lambda x: np.mean(x[1][category]) if x[1][category] else 0)
                best_performers[category] = (best_alg[0], np.mean(best_alg[1][category]))
        
        report.append(f"🏆 Most Trajectories Detected: {best_performers['total_trajectories'][0].upper()} "
                     f"({best_performers['total_trajectories'][1]:.1f} avg)")
        report.append(f"⚡ Fastest Processing: {best_performers['computation_times'][0].upper()} "
                     f"({best_performers['computation_times'][1]:.4f}s avg)")
        report.append(f"🎯 Highest Quality: {best_performers['trajectory_qualities'][0].upper()} "
                     f"({best_performers['trajectory_qualities'][1]:.3f} avg quality)")
        report.append("")
        
        # Detailed Algorithm Performance
        report.append("DETAILED ALGORITHM PERFORMANCE ANALYSIS")
        report.append("=" * 60)
        
        for algorithm_name, stats in algorithm_stats.items():
            if not stats['total_trajectories']:
                continue
            
            avg_trajectories = np.mean(stats['total_trajectories'])
            std_trajectories = np.std(stats['total_trajectories'])
            avg_time = np.mean(stats['computation_times'])
            std_time = np.std(stats['computation_times'])
            avg_quality = np.mean(stats['trajectory_qualities']) if stats['trajectory_qualities'] else 0
            std_quality = np.std(stats['trajectory_qualities']) if stats['trajectory_qualities'] else 0
            avg_length = np.mean(stats['trajectory_lengths']) if stats['trajectory_lengths'] else 0
            avg_frames = np.mean(stats['frames_processed']) if stats['frames_processed'] else 0
            
            # Calculate efficiency and consistency
            efficiency = avg_trajectories / max(avg_time, 1e-6)
            time_consistency = 1.0 / (1.0 + std_time / max(avg_time, 1e-6))
            quality_consistency = 1.0 / (1.0 + std_quality / max(avg_quality, 1e-6)) if avg_quality > 0 else 0
            
            report.append(f"\n{algorithm_name.upper()} ALGORITHM ANALYSIS:")
            report.append("-" * (len(algorithm_name) + 20))
            report.append(f"  Sessions Processed: {stats['sessions_processed']}")
            report.append(f"  Average Trajectories: {avg_trajectories:.2f} ± {std_trajectories:.2f}")
            report.append(f"  Average Processing Time: {avg_time:.4f}s ± {std_time:.4f}s")
            report.append(f"  Average Trajectory Quality: {avg_quality:.3f} ± {std_quality:.3f}")
            report.append(f"  Average Trajectory Length: {avg_length:.1f} frames")
            report.append(f"  Average Frames Processed: {avg_frames:.0f}/session")
            report.append(f"  Processing Efficiency: {efficiency:.1f} trajectories/second")
            report.append(f"  Time Consistency: {time_consistency:.3f}")
            report.append(f"  Quality Consistency: {quality_consistency:.3f}")
            
            # Performance rating
            if efficiency > 10 and avg_quality > 0.7 and time_consistency > 0.8:
                rating = "EXCELLENT"
            elif efficiency > 5 and avg_quality > 0.5 and time_consistency > 0.6:
                rating = "GOOD"
            elif efficiency > 2 and avg_quality > 0.3:
                rating = "FAIR"
            else:
                rating = "NEEDS_IMPROVEMENT"
            
            report.append(f"  Overall Rating: {rating}")
        
        # Subject-wise Performance Analysis
        report.append(f"\n\nSUBJECT-WISE PERFORMANCE SUMMARY")
        report.append("=" * 50)
        
        for subject_id, subject_data in subject_performance.items():
            report.append(f"\nSubject {subject_id}:")
            
            for algorithm_name, perf_data in subject_data.items():
                avg_traj_per_session = perf_data['total_trajectories'] / max(perf_data['sessions'], 1)
                avg_time_per_session = perf_data['total_time'] / max(perf_data['sessions'], 1)
                
                report.append(f"  {algorithm_name}: {avg_traj_per_session:.1f} traj/session, "
                             f"{avg_time_per_session:.3f}s/session")
        
        # Recommendations
        report.append(f"\n\nRECOMMENDATIONS & INSIGHTS")
        report.append("=" * 50)
        
        # Performance-based recommendations
        fastest_alg = best_performers['computation_times'][0]
        most_accurate_alg = best_performers['trajectory_qualities'][0]
        most_sensitive_alg = best_performers['total_trajectories'][0]
        
        report.append("Algorithm Selection Guidelines:")
        report.append(f"• For Real-time Applications: {fastest_alg.upper()} (fastest processing)")
        report.append(f"• For High-precision Analysis: {most_accurate_alg.upper()} (highest quality)")
        report.append(f"• For Maximum Detection: {most_sensitive_alg.upper()} (most trajectories)")
        
        report.append("\nPerformance Optimization Insights:")
        total_sessions = sum(stats['sessions_processed'] for stats in algorithm_stats.values())
        if total_sessions > 0:
            avg_processing_time = np.mean([np.mean(stats['computation_times']) 
                                         for stats in algorithm_stats.values() 
                                         if stats['computation_times']])
            
            report.append(f"• Average processing time across all algorithms: {avg_processing_time:.3f}s")
            report.append(f"• Frame processing efficiency varies by algorithm (see detailed analysis)")
            report.append(f"• Quality-speed trade-off is evident across different algorithms")
        
        # Configuration insights
        report.append(f"\nConfiguration Impact Analysis:")
        report.append(f"• Frame limit setting ({Config.MAX_FRAMES_PER_EPOCH} frames/epoch) affects:")
        report.append(f"  - Processing speed (lower = faster)")
        report.append(f"  - Memory usage (lower = less memory)")
        report.append(f"  - Trajectory completeness (higher = more complete)")
        
        report.append("\nGeneral Insights:")
        report.append("• Algorithm performance may vary significantly with different EEG data characteristics")
        report.append("• Consider data-specific parameter tuning for optimal results")
        report.append("• Multiple algorithm approaches provide robust analysis framework")
        
        return "\n".join(report)
        
    except Exception as e:
        logger.error(f"Enhanced summary report generation failed: {e}")
        return "Enhanced summary report generation failed. Please check logs for details."

def cleanup_memory():
    """Clean up memory"""
    gc.collect()
    
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        if memory_mb > Config.MEMORY_LIMIT_MB:
            logging.getLogger(__name__).warning(
                f"High memory usage: {memory_mb:.1f} MB (limit: {Config.MEMORY_LIMIT_MB} MB)"
            )
            return False
    except ImportError:
        pass
    
    return True

def print_final_summary(all_results, enhanced_comparison_results):
    """Print final experiment summary"""
    print("\n" + "="*80)
    print("ENHANCED ALGORITHM COMPARISON EXPERIMENT SUMMARY")
    print("="*80)
    
    # Basic statistics
    n_subjects = len(all_results)
    total_sessions = sum(len(sessions) for sessions in all_results.values())
    
    print(f"✓ Successfully Processed:")
    print(f"  • Subjects: {n_subjects}")
    print(f"  • Total Sessions: {total_sessions}")
    print(f"  • Frame Limit: {Config.MAX_FRAMES_PER_EPOCH} frames/epoch")
    
    if enhanced_comparison_results:
        metrics = enhanced_comparison_results['comprehensive_metrics']
        print(f"  • Algorithms Compared: {len(metrics)}")
        print(f"  • Algorithm List: {', '.join(metrics.keys())}")
        
        # Show top performers
        if metrics:
            best_overall = max(metrics.items(), key=lambda x: x[1]['composite_performance_score'])
            fastest = min(metrics.items(), key=lambda x: x[1]['avg_computation_times'])
            highest_quality = max(metrics.items(), key=lambda x: x[1]['avg_quality_scores'])
            
            print(f"\n🏆 Top Performers:")
            print(f"  • Best Overall: {best_overall[0].upper()} (Score: {best_overall[1]['composite_performance_score']:.3f})")
            print(f"  • Fastest: {fastest[0].upper()} ({fastest[1]['avg_computation_times']:.4f}s)")
            print(f"  • Highest Quality: {highest_quality[0].upper()} (Score: {highest_quality[1]['avg_quality_scores']:.3f})")
    
    # Output locations
    print(f"\n📂 Results Saved To:")
    print(f"  • Trajectory Plots: {os.path.join(Config.RESULTS_ROOT, 'trajectories')}")
    print(f"  • Algorithm Comparison: {os.path.join(Config.RESULTS_ROOT, 'algorithm_comparison')}")
    
    if enhanced_comparison_results:
        print(f"  • Detailed Report: {enhanced_comparison_results['report_path']}")
        print(f"  • Visualization Suite: {enhanced_comparison_results['visualization_dir']}")
        print(f"  • CSV Data: {enhanced_comparison_results['csv_path']}")
    
    print("="*80)
    print("🎉 Enhanced Algorithm Comparison Experiment Complete!")
    print("="*80)

def main():
    """Main experiment workflow"""
    parser = argparse.ArgumentParser(description='Enhanced EEG Trajectory Analysis with Algorithm Comparison')
    parser.add_argument('--subjects', type=int, default=None, 
                       help='Maximum number of subjects to process')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Maximum epochs per subject')
    parser.add_argument('--frames', type=int, default=None,
                       help='Maximum frames per epoch')
    parser.add_argument('--algorithms', nargs='+', default=None,
                       help='Algorithms to compare', choices=Config.COMPARISON_ALGORITHMS)
    parser.add_argument('--disable-comparison', action='store_true',
                       help='Disable algorithm comparison (use greedy only)')
    parser.add_argument('--fast-mode', action='store_true',
                       help='Enable fast mode (reduced frames and epochs)')
    
    args = parser.parse_args()
    
    # Fast mode configuration
    if args.fast_mode:
        Config.set_max_frames(100, 'epoch')
        Config.MAX_EPOCHS_PER_SUBJECT = 1
        Config.MAX_SUBJECTS = 3
        print("🚀 Fast mode enabled: reduced processing for quick testing")
    
    # Print system information
    print_system_info()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting Enhanced EEG Topography Motion Trajectory Analysis Experiment")
    
    try:
        # Validate configuration
        if not validate_config():
            return 1
        
        # Apply command line arguments
        if args.subjects:
            Config.MAX_SUBJECTS = args.subjects
        if args.epochs:
            Config.MAX_EPOCHS_PER_SUBJECT = args.epochs
        if args.frames:
            Config.set_max_frames(args.frames, 'epoch')
            logger.info(f"Frame limit set to: {args.frames}")
        if args.algorithms:
            Config.COMPARISON_ALGORITHMS = args.algorithms
        if args.disable_comparison:
            Config.ENABLE_ALGORITHM_COMPARISON = False
            Config.COMPARISON_ALGORITHMS = ['greedy']
        
        # Initialize components
        logger.info("Initializing enhanced analysis components...")
        
        data_loader = EEGDataLoader(Config.DATA_ROOT, Config)
        topo_generator = TopographyGenerator(Config)
        analyzer = TrajectoryAnalyzer(Config)
        visualizer = Visualizer(Config)
        
        # Load data
        logger.info("Loading EEG data...")
        all_data = data_loader.load_all_subjects(Config.MAX_SUBJECTS)
        
        if not all_data:
            logger.error("Failed to load any EEG data, please check data path and format")
            print("\n❌ Failed to load any EEG data, please check data path and format")
            return 1
        
        logger.info(f"Successfully loaded data from {len(all_data)} subjects")
        
        # Store all results
        all_results = {}
        
        # Process each subject
        total_subjects = len(all_data)
        processed_subjects = 0
        
        print(f"\n🔄 Processing {total_subjects} subjects with {len(Config.COMPARISON_ALGORITHMS)} algorithms...")
        
        for subject_id, sessions in tqdm(all_data.items(), desc="Processing subjects"):
            try:
                if Config.ENABLE_ALGORITHM_COMPARISON:
                    subject_results = process_subject_with_multiple_algorithms(
                        data_loader, topo_generator, analyzer, visualizer,
                        subject_id, sessions, logger
                    )
                else:
                    logger.info("Using single algorithm mode (greedy)")
                    subject_results = None
                
                if subject_results:
                    all_results[subject_id] = subject_results
                    processed_subjects += 1
                    
                    # Periodic memory cleanup
                    if processed_subjects % 2 == 0:
                        cleanup_memory()
                        logger.info(f"Processed {processed_subjects}/{total_subjects} subjects")
                else:
                    logger.warning(f"Subject {subject_id} produced no valid results")
                    
            except Exception as e:
                logger.error(f"Serious error processing subject {subject_id}: {e}")
                continue
        
        if processed_subjects == 0:
            logger.error("No subject data was successfully processed")
            print("\n❌ No subject data was successfully processed")
            return 1
        
        logger.info(f"Data processing complete, successfully processed {processed_subjects} subjects")
        
        # Generate enhanced algorithm comparison and visualizations
        enhanced_comparison_results = None
        if Config.ENABLE_ALGORITHM_COMPARISON and all_results:
            print("\n📊 Running enhanced algorithm comparison analysis...")
            # 🔧 FIXED: Correct parameter order
            enhanced_comparison_results = run_enhanced_algorithm_comparison(Config, all_results, visualizer)
            
            if enhanced_comparison_results:
                # Create enhanced summary report
                enhanced_report = create_enhanced_summary_report(all_results, logger)
                
                # Save enhanced report
                enhanced_report_path = os.path.join(Config.RESULTS_ROOT, "enhanced_experiment_summary.txt")
                with open(enhanced_report_path, 'w', encoding='utf-8') as f:
                    f.write(enhanced_report)
                
                logger.info(f"Enhanced summary report saved: {enhanced_report_path}")
                
                # Create overall summary visualization
                summary_viz_path = os.path.join(Config.RESULTS_ROOT, "experiment_summary.png")
                visualizer.create_summary_visualization(all_results, summary_viz_path)
        else:
            logger.info("Algorithm comparison disabled or no valid results")
        
        # Save complete results
        results_path = os.path.join(Config.RESULTS_ROOT, "complete_experiment_results.pkl")
        try:
            with open(results_path, 'wb') as f:
                pickle.dump({
                    'experiment_results': all_results,
                    'enhanced_comparison': enhanced_comparison_results,
                    'config_summary': Config.get_config_summary(),
                    'experiment_summary': Config.get_experiment_summary()
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Complete results saved: {results_path}")
        except Exception as e:
            logger.error(f"Failed to save complete results: {e}")
        
        # Print final summary
        print_final_summary(all_results, enhanced_comparison_results)
        
        logger.info("Enhanced algorithm comparison experiment completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Experiment was interrupted by user")
        print("\n🛑 Experiment was interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"Unexpected error during experiment: {e}")
        print(f"\n❌ Unexpected error during experiment: {e}")
        return 1
        
    finally:
        cleanup_memory()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)