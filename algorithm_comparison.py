#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced EEG Trajectory Tracking Algorithm Comparison Module
Comprehensive evaluation and comparison of different tracking algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
import time
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import os

# Set matplotlib to use English fonts only
plt.rcParams['font.family'] = 'DejaVu Sans'

class EnhancedAlgorithmComparison:
    """Enhanced algorithm comparison with better visualization and analysis"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def calculate_comprehensive_metrics(self, algorithm_results: Dict) -> Dict:
        """Calculate comprehensive performance metrics for each algorithm"""
        comprehensive_metrics = {}
        
        for algorithm_name, results in algorithm_results.items():
            metrics = {
                'trajectory_count': [],
                'computation_times': [],
                'trajectory_lengths': [],
                'quality_scores': [],
                'frames_processed': [],
                'efficiency_scores': [],
                'stability_scores': []
            }
            
            # Collect raw data
            for session_data in results.values():
                if isinstance(session_data, dict) and 'trajectories' in session_data:
                    metrics['trajectory_count'].append(len(session_data['trajectories']))
                    metrics['computation_times'].append(session_data.get('total_computation_time', 0))
                    metrics['frames_processed'].append(session_data.get('total_frames_processed', 0))
                    
                    # Extract trajectory-level metrics
                    for traj_data in session_data['trajectories'].values():
                        metrics['trajectory_lengths'].append(traj_data.get('length', 0))
                        metrics['quality_scores'].append(traj_data.get('quality_score', 0))
                    
                    # Calculate efficiency: trajectories per second
                    time_taken = session_data.get('total_computation_time', 1e-6)
                    traj_count = len(session_data['trajectories'])
                    efficiency = traj_count / max(time_taken, 1e-6)
                    metrics['efficiency_scores'].append(efficiency)
            
            # Calculate summary statistics
            summary = {}
            for metric_name, values in metrics.items():
                if values:
                    summary[f'avg_{metric_name}'] = np.mean(values)
                    summary[f'std_{metric_name}'] = np.std(values)
                    summary[f'min_{metric_name}'] = np.min(values)
                    summary[f'max_{metric_name}'] = np.max(values)
                    summary[f'median_{metric_name}'] = np.median(values)
                else:
                    summary[f'avg_{metric_name}'] = 0
                    summary[f'std_{metric_name}'] = 0
                    summary[f'min_{metric_name}'] = 0
                    summary[f'max_{metric_name}'] = 0
                    summary[f'median_{metric_name}'] = 0
            
            # Calculate composite performance score
            traj_score = min(1.0, summary['avg_trajectory_count'] / 5.0)
            quality_score = summary['avg_quality_scores']
            efficiency_score = min(1.0, summary['avg_efficiency_scores'] / 10.0)
            stability_score = 1.0 / (1.0 + summary['std_computation_times'] / max(summary['avg_computation_times'], 1e-6))
            
            composite_score = (traj_score * 0.3 + quality_score * 0.3 + 
                             efficiency_score * 0.25 + stability_score * 0.15)
            
            summary['composite_performance_score'] = composite_score
            summary['raw_data'] = metrics
            
            comprehensive_metrics[algorithm_name] = summary
        
        return comprehensive_metrics
    
    def generate_detailed_report(self, comprehensive_metrics: Dict) -> str:
        """Generate detailed comparison report with actionable insights"""
        report = []
        report.append("=" * 80)
        report.append("ENHANCED EEG TRAJECTORY TRACKING ALGORITHM COMPARISON REPORT")
        report.append("=" * 80)
        report.append(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Algorithms Analyzed: {len(comprehensive_metrics)}")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        
        # Find best performers in each category
        best_overall = max(comprehensive_metrics.items(), 
                          key=lambda x: x[1]['composite_performance_score'])
        best_speed = min(comprehensive_metrics.items(), 
                        key=lambda x: x[1]['avg_computation_times'])
        best_quality = max(comprehensive_metrics.items(), 
                          key=lambda x: x[1]['avg_quality_scores'])
        best_efficiency = max(comprehensive_metrics.items(), 
                             key=lambda x: x[1]['avg_efficiency_scores'])
        
        report.append(f"• Best Overall Performance: {best_overall[0].upper()} "
                     f"(Score: {best_overall[1]['composite_performance_score']:.3f})")
        report.append(f"• Fastest Algorithm: {best_speed[0].upper()} "
                     f"({best_speed[1]['avg_computation_times']:.4f}s avg)")
        report.append(f"• Highest Quality: {best_quality[0].upper()} "
                     f"(Score: {best_quality[1]['avg_quality_scores']:.3f})")
        report.append(f"• Most Efficient: {best_efficiency[0].upper()} "
                     f"({best_efficiency[1]['avg_efficiency_scores']:.1f} traj/s)")
        report.append("")
        
        # Detailed Algorithm Analysis
        report.append("DETAILED ALGORITHM ANALYSIS")
        report.append("-" * 50)
        
        for algorithm_name, metrics in comprehensive_metrics.items():
            report.append(f"\n{algorithm_name.upper()} ALGORITHM:")
            report.append("=" * (len(algorithm_name) + 11))
            
            # Performance Metrics
            report.append("Performance Metrics:")
            report.append(f"  • Average Trajectories Detected: {metrics['avg_trajectory_count']:.2f} ± {metrics['std_trajectory_count']:.2f}")
            report.append(f"  • Average Computation Time: {metrics['avg_computation_times']:.4f}s ± {metrics['std_computation_times']:.4f}s")
            report.append(f"  • Average Trajectory Quality: {metrics['avg_quality_scores']:.3f} ± {metrics['std_quality_scores']:.3f}")
            report.append(f"  • Average Trajectory Length: {metrics['avg_trajectory_lengths']:.1f} ± {metrics['std_trajectory_lengths']:.1f} frames")
            report.append(f"  • Processing Efficiency: {metrics['avg_efficiency_scores']:.1f} trajectories/second")
            report.append(f"  • Composite Performance Score: {metrics['composite_performance_score']:.3f}")
            
            # Reliability Metrics
            report.append("\nReliability Analysis:")
            cv_time = metrics['std_computation_times'] / max(metrics['avg_computation_times'], 1e-6)
            cv_quality = metrics['std_quality_scores'] / max(metrics['avg_quality_scores'], 1e-6)
            
            report.append(f"  • Time Consistency (CV): {cv_time:.3f} {'(Excellent)' if cv_time < 0.1 else '(Good)' if cv_time < 0.3 else '(Poor)'}")
            report.append(f"  • Quality Consistency (CV): {cv_quality:.3f} {'(Excellent)' if cv_quality < 0.1 else '(Good)' if cv_quality < 0.3 else '(Poor)'}")
            
            # Performance Range
            report.append("\nPerformance Range:")
            report.append(f"  • Trajectory Count Range: {metrics['min_trajectory_count']:.0f} - {metrics['max_trajectory_count']:.0f}")
            report.append(f"  • Time Range: {metrics['min_computation_times']:.4f}s - {metrics['max_computation_times']:.4f}s")
            report.append(f"  • Quality Range: {metrics['min_quality_scores']:.3f} - {metrics['max_quality_scores']:.3f}")
        
        # Comparative Analysis
        report.append("\n\nCOMPARATIVE ANALYSIS")
        report.append("-" * 50)
        
        # Statistical significance (simplified)
        report.append("Performance Ranking by Category:")
        
        categories = [
            ('Overall Performance', 'composite_performance_score'),
            ('Speed', 'avg_computation_times', True),  # Lower is better
            ('Quality', 'avg_quality_scores'),
            ('Efficiency', 'avg_efficiency_scores'),
            ('Trajectory Count', 'avg_trajectory_count')
        ]
        
        for category_name, metric_key, *reverse in categories:
            is_reverse = len(reverse) > 0 and reverse[0]
            sorted_algorithms = sorted(comprehensive_metrics.items(), 
                                     key=lambda x: x[1][metric_key], 
                                     reverse=not is_reverse)
            
            report.append(f"\n{category_name}:")
            for i, (alg_name, metrics) in enumerate(sorted_algorithms, 1):
                value = metrics[metric_key]
                if 'time' in metric_key:
                    report.append(f"  {i}. {alg_name.upper()}: {value:.4f}s")
                elif 'score' in metric_key:
                    report.append(f"  {i}. {alg_name.upper()}: {value:.3f}")
                else:
                    report.append(f"  {i}. {alg_name.upper()}: {value:.2f}")
        
        # Recommendations
        report.append("\n\nRECOMMENDATIONS")
        report.append("-" * 40)
        
        report.append("Use Case Recommendations:")
        report.append("• Real-time Processing: Choose the fastest algorithm with acceptable quality")
        report.append("• High-precision Analysis: Choose the highest quality algorithm")
        report.append("• Resource-constrained Environments: Choose the most efficient algorithm")
        report.append("• Batch Processing: Choose the best overall performance algorithm")
        
        report.append("\nSpecific Recommendations:")
        for alg_name, metrics in comprehensive_metrics.items():
            score = metrics['composite_performance_score']
            speed = metrics['avg_computation_times']
            quality = metrics['avg_quality_scores']
            
            if score > 0.7:
                recommendation = "Excellent for most applications"
            elif speed < 0.1 and quality > 0.6:
                recommendation = "Good for real-time applications"
            elif quality > 0.8:
                recommendation = "Ideal for high-precision tasks"
            elif metrics['avg_efficiency_scores'] > 15:
                recommendation = "Best for high-throughput scenarios"
            else:
                recommendation = "Suitable for basic applications"
            
            report.append(f"• {alg_name.upper()}: {recommendation}")
        
        # Technical Notes
        report.append("\n\nTECHNICAL NOTES")
        report.append("-" * 40)
        report.append("• Performance scores are normalized to 0-1 scale")
        report.append("• Composite score weights: Trajectories(30%), Quality(30%), Efficiency(25%), Stability(15%)")
        report.append("• Coefficient of Variation (CV) indicates consistency: <0.1=Excellent, 0.1-0.3=Good, >0.3=Poor")
        report.append("• Results may vary with different data characteristics and parameters")
        
        return "\n".join(report)
    
    def create_comprehensive_visualizations(self, comprehensive_metrics: Dict, 
                                          save_dir: str, visualizer):
        """Create comprehensive visualization suite"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Main comparison chart
        main_comparison_path = os.path.join(save_dir, "main_algorithm_comparison.png")
        visualizer.create_algorithm_comparison_plot(comprehensive_metrics, main_comparison_path)
        
        # 2. Performance radar chart  
        radar_path = os.path.join(save_dir, "performance_radar_chart.png")
        visualizer.create_performance_radar_chart(comprehensive_metrics, radar_path)
        
        # 3. Detailed comparison table
        table_path = os.path.join(save_dir, "detailed_comparison_table.png")
        visualizer.create_detailed_comparison_table(comprehensive_metrics, table_path)
        
        # 4. Statistical analysis plots
        self._create_statistical_plots(comprehensive_metrics, save_dir)
        
        # 5. Performance trends
        self._create_performance_trends(comprehensive_metrics, save_dir)
        
        self.logger.info(f"Comprehensive visualizations saved to {save_dir}")
    
    def _create_statistical_plots(self, comprehensive_metrics: Dict, save_dir: str):
        """Create statistical analysis plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Statistical Performance Analysis', fontsize=16, fontweight='bold')
            
            algorithms = list(comprehensive_metrics.keys())
            
            # 1. Box plot for trajectory counts
            ax = axes[0, 0]
            traj_data = []
            labels = []
            
            for alg in algorithms:
                raw_data = comprehensive_metrics[alg]['raw_data']
                if raw_data['trajectory_count']:
                    traj_data.append(raw_data['trajectory_count'])
                    labels.append(alg)
            
            if traj_data:
                bp = ax.boxplot(traj_data, labels=labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], plt.cm.Set1(np.linspace(0, 1, len(labels)))):
                    patch.set_facecolor(color)
                ax.set_title('Trajectory Count Distribution')
                ax.set_ylabel('Number of Trajectories')
                ax.tick_params(axis='x', rotation=45)
            
            # 2. Computation time variability
            ax = axes[0, 1]
            time_data = []
            time_labels = []
            
            for alg in algorithms:
                raw_data = comprehensive_metrics[alg]['raw_data']
                if raw_data['computation_times']:
                    time_data.append(raw_data['computation_times'])
                    time_labels.append(alg)
            
            if time_data:
                bp = ax.boxplot(time_data, labels=time_labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], plt.cm.Set1(np.linspace(0, 1, len(time_labels)))):
                    patch.set_facecolor(color)
                ax.set_title('Computation Time Distribution')
                ax.set_ylabel('Time (seconds)')
                ax.tick_params(axis='x', rotation=45)
            
            # 3. Quality score histogram
            ax = axes[1, 0]
            for i, alg in enumerate(algorithms):
                raw_data = comprehensive_metrics[alg]['raw_data']
                if raw_data['quality_scores']:
                    ax.hist(raw_data['quality_scores'], alpha=0.7, 
                           label=alg, bins=20, density=True)
            
            ax.set_title('Quality Score Distribution')
            ax.set_xlabel('Quality Score')
            ax.set_ylabel('Density')
            ax.legend()
            
            # 4. Efficiency comparison
            ax = axes[1, 1]
            efficiencies = []
            eff_labels = []
            
            for alg in algorithms:
                eff = comprehensive_metrics[alg]['avg_efficiency_scores']
                efficiencies.append(eff)
                eff_labels.append(alg)
            
            bars = ax.bar(eff_labels, efficiencies, 
                         color=plt.cm.Set1(np.linspace(0, 1, len(eff_labels))), alpha=0.7)
            ax.set_title('Processing Efficiency Comparison')
            ax.set_ylabel('Trajectories per Second')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, eff in zip(bars, efficiencies):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{eff:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            save_path = os.path.join(save_dir, "statistical_analysis.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Statistical plots saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create statistical plots: {e}")
            if 'fig' in locals():
                plt.close(fig)
    
    def _create_performance_trends(self, comprehensive_metrics: Dict, save_dir: str):
        """Create performance trend analysis"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Performance Trends Analysis', fontsize=16, fontweight='bold')
            
            algorithms = list(comprehensive_metrics.keys())
            colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
            
            # 1. Performance vs Speed Trade-off
            ax = axes[0, 0]
            for i, alg in enumerate(algorithms):
                metrics = comprehensive_metrics[alg]
                x = metrics['avg_computation_times']
                y = metrics['composite_performance_score']
                ax.scatter(x, y, s=100, c=[colors[i]], alpha=0.7, label=alg)
                ax.annotate(alg, (x, y), xytext=(5, 5), textcoords='offset points')
            
            ax.set_xlabel('Average Computation Time (s)')
            ax.set_ylabel('Composite Performance Score')
            ax.set_title('Performance vs Speed Trade-off')
            ax.grid(True, alpha=0.3)
            
            # 2. Quality vs Efficiency
            ax = axes[0, 1]
            for i, alg in enumerate(algorithms):
                metrics = comprehensive_metrics[alg]
                x = metrics['avg_efficiency_scores']
                y = metrics['avg_quality_scores']
                ax.scatter(x, y, s=100, c=[colors[i]], alpha=0.7, label=alg)
                ax.annotate(alg, (x, y), xytext=(5, 5), textcoords='offset points')
            
            ax.set_xlabel('Efficiency (trajectories/s)')
            ax.set_ylabel('Average Quality Score')
            ax.set_title('Quality vs Efficiency Trade-off')
            ax.grid(True, alpha=0.3)
            
            # 3. Consistency Analysis
            ax = axes[1, 0]
            consistency_metrics = []
            for alg in algorithms:
                metrics = comprehensive_metrics[alg]
                time_cv = metrics['std_computation_times'] / max(metrics['avg_computation_times'], 1e-6)
                quality_cv = metrics['std_quality_scores'] / max(metrics['avg_quality_scores'], 1e-6)
                consistency_score = 1.0 / (1.0 + time_cv + quality_cv)
                consistency_metrics.append(consistency_score)
            
            bars = ax.bar(algorithms, consistency_metrics, color=colors, alpha=0.7)
            ax.set_title('Algorithm Consistency Score')
            ax.set_ylabel('Consistency Score (higher is better)')
            ax.tick_params(axis='x', rotation=45)
            
            for bar, score in zip(bars, consistency_metrics):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
            
            # 4. Overall Ranking
            ax = axes[1, 1]
            ranking_categories = ['Speed', 'Quality', 'Efficiency', 'Consistency']
            
            # Calculate rankings for each category
            speed_ranking = {alg: i+1 for i, (alg, _) in enumerate(
                sorted(comprehensive_metrics.items(), key=lambda x: x[1]['avg_computation_times']))}
            quality_ranking = {alg: i+1 for i, (alg, _) in enumerate(
                sorted(comprehensive_metrics.items(), key=lambda x: x[1]['avg_quality_scores'], reverse=True))}
            efficiency_ranking = {alg: i+1 for i, (alg, _) in enumerate(
                sorted(comprehensive_metrics.items(), key=lambda x: x[1]['avg_efficiency_scores'], reverse=True))}
            consistency_ranking = {alg: i+1 for i, (alg, score) in enumerate(
                sorted(zip(algorithms, consistency_metrics), key=lambda x: x[1], reverse=True))}
            
            # Create heatmap data
            ranking_data = []
            for alg in algorithms:
                ranking_data.append([
                    speed_ranking[alg],
                    quality_ranking[alg], 
                    efficiency_ranking[alg],
                    consistency_ranking[alg]
                ])
            
            im = ax.imshow(ranking_data, cmap='RdYlGn_r', aspect='auto')
            ax.set_xticks(range(len(ranking_categories)))
            ax.set_xticklabels(ranking_categories)
            ax.set_yticks(range(len(algorithms)))
            ax.set_yticklabels(algorithms)
            ax.set_title('Algorithm Ranking Heatmap (1=Best)')
            
            # Add text annotations
            for i in range(len(algorithms)):
                for j in range(len(ranking_categories)):
                    text = ax.text(j, i, ranking_data[i][j], ha="center", va="center", color="black")
            
            plt.colorbar(im, ax=ax)
            
            plt.tight_layout()
            
            save_path = os.path.join(save_dir, "performance_trends.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Performance trends saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create performance trends: {e}")
            if 'fig' in locals():
                plt.close(fig)
    
    def export_results_to_csv(self, comprehensive_metrics: Dict, save_path: str):
        """Export detailed results to CSV for further analysis"""
        try:
            data_rows = []
            
            for algorithm_name, metrics in comprehensive_metrics.items():
                row = {
                    'Algorithm': algorithm_name,
                    'Avg_Trajectories': metrics['avg_trajectory_count'],
                    'Std_Trajectories': metrics['std_trajectory_count'],
                    'Avg_Computation_Time': metrics['avg_computation_times'],
                    'Std_Computation_Time': metrics['std_computation_times'],
                    'Avg_Quality': metrics['avg_quality_scores'],
                    'Std_Quality': metrics['std_quality_scores'],
                    'Avg_Length': metrics['avg_trajectory_lengths'],
                    'Std_Length': metrics['std_trajectory_lengths'],
                    'Avg_Efficiency': metrics['avg_efficiency_scores'],
                    'Composite_Score': metrics['composite_performance_score'],
                    'Min_Trajectories': metrics['min_trajectory_count'],
                    'Max_Trajectories': metrics['max_trajectory_count'],
                    'Min_Time': metrics['min_computation_times'],
                    'Max_Time': metrics['max_computation_times'],
                    'Min_Quality': metrics['min_quality_scores'],
                    'Max_Quality': metrics['max_quality_scores']
                }
                data_rows.append(row)
            
            df = pd.DataFrame(data_rows)
            df.to_csv(save_path, index=False)
            
            self.logger.info(f"Results exported to CSV: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export results to CSV: {e}")


def run_enhanced_algorithm_comparison(config, all_results, visualizer):
    """Run enhanced algorithm comparison with improved analysis"""
    comparison = EnhancedAlgorithmComparison(config)
    
    print("\n" + "="*60)
    print("Running Enhanced Algorithm Comparison Analysis...")
    print("="*60)
    
    try:
        # Calculate comprehensive metrics
        print("📊 Calculating comprehensive performance metrics...")
        comprehensive_metrics = comparison.calculate_comprehensive_metrics(all_results)
        
        if not comprehensive_metrics:
            print("❌ No algorithm results available for comparison")
            return None
        
        print(f"✓ Analyzed {len(comprehensive_metrics)} algorithms")
        
        # Generate detailed report
        print("📝 Generating detailed analysis report...")
        detailed_report = comparison.generate_detailed_report(comprehensive_metrics)
        
        # Save report
        comparison_dir = os.path.join(config.RESULTS_ROOT, "algorithm_comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        report_path = os.path.join(comparison_dir, "enhanced_comparison_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(detailed_report)
        
        print(f"✓ Detailed report saved: {report_path}")
        
        # Create comprehensive visualizations
        print("📈 Creating comprehensive visualizations...")
        comparison.create_comprehensive_visualizations(
            comprehensive_metrics, comparison_dir, visualizer)
        
        # Export to CSV for further analysis
        csv_path = os.path.join(comparison_dir, "algorithm_metrics.csv")
        comparison.export_results_to_csv(comprehensive_metrics, csv_path)
        
        # Display summary
        print("\n" + "="*60)
        print("ALGORITHM COMPARISON SUMMARY")
        print("="*60)
        
        # Find best performers
        best_overall = max(comprehensive_metrics.items(), 
                          key=lambda x: x[1]['composite_performance_score'])
        fastest = min(comprehensive_metrics.items(), 
                     key=lambda x: x[1]['avg_computation_times'])
        highest_quality = max(comprehensive_metrics.items(), 
                            key=lambda x: x[1]['avg_quality_scores'])
        
        print(f"🏆 Best Overall: {best_overall[0].upper()} (Score: {best_overall[1]['composite_performance_score']:.3f})")
        print(f"⚡ Fastest: {fastest[0].upper()} ({fastest[1]['avg_computation_times']:.4f}s)")
        print(f"🎯 Highest Quality: {highest_quality[0].upper()} (Score: {highest_quality[1]['avg_quality_scores']:.3f})")
        
        print(f"\n📂 Results Location:")
        print(f"   • Report: {report_path}")
        print(f"   • Visualizations: {comparison_dir}")
        print(f"   • CSV Data: {csv_path}")
        
        print("\n" + "="*60)
        print("Enhanced Algorithm Comparison Complete! 🎉")
        print("="*60)
        
        return {
            'comprehensive_metrics': comprehensive_metrics,
            'report_path': report_path,
            'visualization_dir': comparison_dir,
            'csv_path': csv_path
        }
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Enhanced algorithm comparison failed: {e}")
        print(f"❌ Algorithm comparison failed: {e}")
        return None