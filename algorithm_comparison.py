#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG轨迹跟踪算法对比模块
系统性评估和对比不同的跟踪算法性能
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

class TrackingAlgorithmComparison:
    """跟踪算法对比类"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def hungarian_matching(self, current_regions: List[Dict], 
                          tracked_centers: np.ndarray,
                          distance_threshold: float = 20.0) -> List[Tuple[int, int]]:
        """匈牙利算法匹配"""
        if not current_regions or len(tracked_centers) == 0:
            return []
        
        current_centers = np.array([r['center'] for r in current_regions])
        distances = cdist(tracked_centers, current_centers)
        
        # 使用匈牙利算法求解最优分配
        row_indices, col_indices = linear_sum_assignment(distances)
        
        matches = []
        for row_idx, col_idx in zip(row_indices, col_indices):
            if distances[row_idx, col_idx] < distance_threshold:
                matches.append((row_idx, col_idx))
        
        return matches
    
    def greedy_matching(self, current_regions: List[Dict], 
                       tracked_centers: np.ndarray,
                       distance_threshold: float = 20.0) -> List[Tuple[int, int]]:
        """贪婪算法匹配（当前使用的方法）"""
        if not current_regions or len(tracked_centers) == 0:
            return []
        
        current_centers = np.array([r['center'] for r in current_regions])
        distances = cdist(tracked_centers, current_centers)
        
        matches = []
        used_current = set()
        used_tracked = set()
        
        dist_indices = np.unravel_index(np.argsort(distances.ravel()), distances.shape)
        
        for tracked_idx, current_idx in zip(dist_indices[0], dist_indices[1]):
            if tracked_idx in used_tracked or current_idx in used_current:
                continue
            if distances[tracked_idx, current_idx] < distance_threshold:
                matches.append((tracked_idx, current_idx))
                used_tracked.add(tracked_idx)
                used_current.add(current_idx)
        
        return matches
    
    def kalman_prediction_matching(self, current_regions: List[Dict],
                                  tracked_regions: List,
                                  distance_threshold: float = 20.0) -> List[Tuple[int, int]]:
        """基于卡尔曼滤波预测的匹配"""
        if not current_regions or not tracked_regions:
            return []
        
        # 简化的卡尔曼预测：基于速度的线性预测
        predicted_centers = []
        for region in tracked_regions:
            trajectory = region.trajectory
            if len(trajectory) >= 2:
                # 计算速度
                velocity = np.array(trajectory[-1]) - np.array(trajectory[-2])
                # 预测下一个位置
                predicted_pos = np.array(trajectory[-1]) + velocity
                predicted_centers.append(predicted_pos)
            else:
                predicted_centers.append(trajectory[-1])
        
        if not predicted_centers:
            return []
        
        predicted_centers = np.array(predicted_centers)
        current_centers = np.array([r['center'] for r in current_regions])
        distances = cdist(predicted_centers, current_centers)
        
        # 使用匈牙利算法
        row_indices, col_indices = linear_sum_assignment(distances)
        
        matches = []
        for row_idx, col_idx in zip(row_indices, col_indices):
            if distances[row_idx, col_idx] < distance_threshold:
                matches.append((row_idx, col_idx))
        
        return matches
    
    def overlap_based_matching(self, current_regions: List[Dict],
                              previous_regions: List[Dict],
                              overlap_threshold: float = 0.3) -> List[Tuple[int, int]]:
        """基于区域重叠的匹配"""
        if not current_regions or not previous_regions:
            return []
        
        matches = []
        
        for i, curr_region in enumerate(current_regions):
            curr_mask = curr_region['mask']
            best_match = -1
            best_overlap = 0
            
            for j, prev_region in enumerate(previous_regions):
                prev_mask = prev_region['mask']
                
                # 计算重叠率
                intersection = np.sum(curr_mask & prev_mask)
                union = np.sum(curr_mask | prev_mask)
                
                if union > 0:
                    overlap = intersection / union
                    if overlap > overlap_threshold and overlap > best_overlap:
                        best_overlap = overlap
                        best_match = j
            
            if best_match >= 0:
                matches.append((best_match, i))
        
        return matches

class SimilarityAlgorithmComparison:
    """相似性算法对比类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def dtw_distance(self, traj1: np.ndarray, traj2: np.ndarray) -> float:
        """DTW距离"""
        try:
            from fastdtw import fastdtw
            from scipy.spatial.distance import euclidean
            distance, _ = fastdtw(traj1, traj2, dist=euclidean)
            return distance
        except ImportError:
            return self.euclidean_distance(traj1, traj2)
    
    def euclidean_distance(self, traj1: np.ndarray, traj2: np.ndarray) -> float:
        """欧几里得距离（插值对齐）"""
        # 插值到相同长度
        target_length = min(len(traj1), len(traj2), 50)
        
        if len(traj1) < 2 or len(traj2) < 2:
            return np.linalg.norm(traj1[-1] - traj2[-1])
        
        from scipy.interpolate import interp1d
        
        # 插值
        t1 = np.linspace(0, 1, len(traj1))
        t2 = np.linspace(0, 1, len(traj2))
        t_new = np.linspace(0, 1, target_length)
        
        interp1_x = interp1d(t1, traj1[:, 0], kind='linear')
        interp1_y = interp1d(t1, traj1[:, 1], kind='linear')
        interp2_x = interp1d(t2, traj2[:, 0], kind='linear')
        interp2_y = interp1d(t2, traj2[:, 1], kind='linear')
        
        traj1_interp = np.column_stack([interp1_x(t_new), interp1_y(t_new)])
        traj2_interp = np.column_stack([interp2_x(t_new), interp2_y(t_new)])
        
        return np.linalg.norm(traj1_interp - traj2_interp)
    
    def frechet_distance(self, traj1: np.ndarray, traj2: np.ndarray) -> float:
        """简化的Fréchet距离"""
        # 这里实现简化版本，完整版本需要更复杂的算法
        return self.euclidean_distance(traj1, traj2)
    
    def hausdorff_distance(self, traj1: np.ndarray, traj2: np.ndarray) -> float:
        """Hausdorff距离"""
        # 计算有向Hausdorff距离
        def directed_hausdorff(X, Y):
            return max(min(np.linalg.norm(x - y) for y in Y) for x in X)
        
        return max(directed_hausdorff(traj1, traj2), directed_hausdorff(traj2, traj1))

class ClusteringAlgorithmComparison:
    """聚类算法对比类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def compare_clustering_methods(self, features: np.ndarray, 
                                  true_labels: Optional[np.ndarray] = None) -> Dict:
        """对比不同聚类方法"""
        results = {}
        
        # 确定聚类数量范围
        n_samples = len(features)
        max_clusters = min(n_samples // 2, 8)
        
        if max_clusters < 2:
            return results
        
        # 1. K-means聚类
        for n_clusters in range(2, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
                
                # 计算轮廓系数
                if len(np.unique(labels)) > 1:
                    silhouette = silhouette_score(features, labels)
                else:
                    silhouette = -1
                
                results[f'kmeans_k{n_clusters}'] = {
                    'labels': labels,
                    'silhouette_score': silhouette,
                    'n_clusters': n_clusters,
                    'method': 'kmeans'
                }
                
                # 如果有真实标签，计算ARI
                if true_labels is not None:
                    ari = adjusted_rand_score(true_labels, labels)
                    results[f'kmeans_k{n_clusters}']['ari'] = ari
                    
            except Exception as e:
                self.logger.warning(f"K-means with k={n_clusters} failed: {e}")
        
        # 2. 层次聚类
        for n_clusters in range(2, max_clusters + 1):
            try:
                agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                labels = agg_clustering.fit_predict(features)
                
                if len(np.unique(labels)) > 1:
                    silhouette = silhouette_score(features, labels)
                else:
                    silhouette = -1
                
                results[f'hierarchical_k{n_clusters}'] = {
                    'labels': labels,
                    'silhouette_score': silhouette,
                    'n_clusters': n_clusters,
                    'method': 'hierarchical'
                }
                
                if true_labels is not None:
                    ari = adjusted_rand_score(true_labels, labels)
                    results[f'hierarchical_k{n_clusters}']['ari'] = ari
                    
            except Exception as e:
                self.logger.warning(f"Hierarchical clustering with k={n_clusters} failed: {e}")
        
        # 3. DBSCAN聚类
        eps_values = [0.3, 0.5, 0.7, 1.0]
        min_samples_values = [2, 3, 4]
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(features)
                    
                    n_clusters = len(np.unique(labels[labels >= 0]))
                    
                    if n_clusters > 1:
                        # 只对非噪声点计算轮廓系数
                        mask = labels >= 0
                        if np.sum(mask) > 1:
                            silhouette = silhouette_score(features[mask], labels[mask])
                        else:
                            silhouette = -1
                    else:
                        silhouette = -1
                    
                    results[f'dbscan_eps{eps}_min{min_samples}'] = {
                        'labels': labels,
                        'silhouette_score': silhouette,
                        'n_clusters': n_clusters,
                        'eps': eps,
                        'min_samples': min_samples,
                        'method': 'dbscan',
                        'n_noise': np.sum(labels == -1)
                    }
                    
                    if true_labels is not None:
                        ari = adjusted_rand_score(true_labels, labels)
                        results[f'dbscan_eps{eps}_min{min_samples}']['ari'] = ari
                        
                except Exception as e:
                    self.logger.warning(f"DBSCAN (eps={eps}, min_samples={min_samples}) failed: {e}")
        
        return results

class AlgorithmBenchmark:
    """算法性能基准测试"""
    
    def __init__(self, config):
        self.config = config
        self.tracking_comparison = TrackingAlgorithmComparison(config)
        self.similarity_comparison = SimilarityAlgorithmComparison()
        self.clustering_comparison = ClusteringAlgorithmComparison()
        self.logger = logging.getLogger(__name__)
    
    def benchmark_tracking_algorithms(self, topographies: np.ndarray) -> Dict:
        """基准测试跟踪算法"""
        results = {
            'hungarian': {'execution_times': [], 'match_counts': []},
            'greedy': {'execution_times': [], 'match_counts': []},
            'kalman': {'execution_times': [], 'match_counts': []},
            'overlap': {'execution_times': [], 'match_counts': []}
        }
        
        self.logger.info("开始跟踪算法基准测试...")
        
        # 模拟跟踪过程
        for frame_idx in range(1, min(50, topographies.shape[0])):  # 限制帧数以节省时间
            # 检测当前帧区域（使用简化的检测逻辑）
            current_topo = topographies[frame_idx]
            prev_topo = topographies[frame_idx - 1]
            
            # 简化的区域检测
            threshold = np.percentile(current_topo[current_topo > 0], 90)
            current_binary = current_topo > threshold
            
            threshold_prev = np.percentile(prev_topo[prev_topo > 0], 90)
            prev_binary = prev_topo > threshold_prev
            
            # 模拟区域
            current_regions = [{'center': (50, 50), 'mask': current_binary}]
            prev_regions = [{'center': (48, 52), 'mask': prev_binary}]
            tracked_centers = np.array([[48, 52]])
            
            # 测试不同算法
            algorithms = {
                'hungarian': lambda: self.tracking_comparison.hungarian_matching(
                    current_regions, tracked_centers),
                'greedy': lambda: self.tracking_comparison.greedy_matching(
                    current_regions, tracked_centers),
                'overlap': lambda: self.tracking_comparison.overlap_based_matching(
                    current_regions, prev_regions)
            }
            
            for alg_name, alg_func in algorithms.items():
                try:
                    start_time = time.time()
                    matches = alg_func()
                    end_time = time.time()
                    
                    results[alg_name]['execution_times'].append(end_time - start_time)
                    results[alg_name]['match_counts'].append(len(matches))
                except Exception as e:
                    self.logger.warning(f"Algorithm {alg_name} failed on frame {frame_idx}: {e}")
        
        return results
    
    def benchmark_similarity_algorithms(self, trajectories: Dict) -> Dict:
        """基准测试相似性算法"""
        if len(trajectories) < 2:
            return {}
        
        results = {
            'dtw': {'execution_times': [], 'distances': []},
            'euclidean': {'execution_times': [], 'distances': []},
            'hausdorff': {'execution_times': [], 'distances': []}
        }
        
        trajectory_list = list(trajectories.values())
        
        self.logger.info("开始相似性算法基准测试...")
        
        # 比较所有轨迹对
        for i in range(min(10, len(trajectory_list))):  # 限制比较数量
            for j in range(i + 1, min(10, len(trajectory_list))):
                traj1 = trajectory_list[i]['trajectory']
                traj2 = trajectory_list[j]['trajectory']
                
                if len(traj1) < 2 or len(traj2) < 2:
                    continue
                
                # 测试不同相似性算法
                algorithms = {
                    'dtw': self.similarity_comparison.dtw_distance,
                    'euclidean': self.similarity_comparison.euclidean_distance,
                    'hausdorff': self.similarity_comparison.hausdorff_distance
                }
                
                for alg_name, alg_func in algorithms.items():
                    try:
                        start_time = time.time()
                        distance = alg_func(traj1, traj2)
                        end_time = time.time()
                        
                        results[alg_name]['execution_times'].append(end_time - start_time)
                        results[alg_name]['distances'].append(distance)
                    except Exception as e:
                        self.logger.warning(f"Similarity algorithm {alg_name} failed: {e}")
        
        return results
    
    def generate_comparison_report(self, tracking_results: Dict, 
                                 similarity_results: Dict,
                                 clustering_results: Dict) -> str:
        """生成算法对比报告"""
        report = []
        report.append("=" * 60)
        report.append("EEG轨迹跟踪算法性能对比报告")
        report.append("=" * 60)
        report.append("")
        
        # 跟踪算法对比
        if tracking_results:
            report.append("1. 跟踪算法性能对比")
            report.append("-" * 30)
            
            for alg_name, metrics in tracking_results.items():
                if metrics['execution_times']:
                    avg_time = np.mean(metrics['execution_times'])
                    avg_matches = np.mean(metrics['match_counts'])
                    
                    report.append(f"{alg_name.upper()}算法:")
                    report.append(f"  平均执行时间: {avg_time*1000:.3f} ms")
                    report.append(f"  平均匹配数量: {avg_matches:.1f}")
                    report.append("")
        
        # 相似性算法对比
        if similarity_results:
            report.append("2. 相似性算法性能对比")
            report.append("-" * 30)
            
            for alg_name, metrics in similarity_results.items():
                if metrics['execution_times']:
                    avg_time = np.mean(metrics['execution_times'])
                    avg_distance = np.mean(metrics['distances'])
                    
                    report.append(f"{alg_name.upper()}算法:")
                    report.append(f"  平均执行时间: {avg_time*1000:.3f} ms")
                    report.append(f"  平均距离: {avg_distance:.3f}")
                    report.append("")
        
        # 聚类算法对比
        if clustering_results:
            report.append("3. 聚类算法性能对比")
            report.append("-" * 30)
            
            best_method = None
            best_score = -1
            
            for method_name, result in clustering_results.items():
                silhouette = result.get('silhouette_score', -1)
                if silhouette > best_score:
                    best_score = silhouette
                    best_method = method_name
                
                report.append(f"{method_name}:")
                report.append(f"  轮廓系数: {silhouette:.3f}")
                report.append(f"  聚类数量: {result.get('n_clusters', 0)}")
                
                if 'ari' in result:
                    report.append(f"  调整兰德指数: {result['ari']:.3f}")
                
                report.append("")
            
            if best_method:
                report.append(f"推荐聚类方法: {best_method} (轮廓系数: {best_score:.3f})")
                report.append("")
        
        report.append("4. 算法选择建议")
        report.append("-" * 30)
        report.append("• 实时应用: 选择执行时间最短的算法")
        report.append("• 精度优先: 选择匹配质量最好的算法")
        report.append("• 平衡方案: 在性能和精度间找到最佳平衡")
        report.append("")
        
        return "\n".join(report)
    
    def visualize_comparison_results(self, results: Dict, save_path: str):
        """可视化对比结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('算法性能对比结果', fontsize=16, fontweight='bold')
        
        # 跟踪算法执行时间对比
        if 'tracking' in results:
            ax = axes[0, 0]
            tracking_data = results['tracking']
            
            alg_names = []
            avg_times = []
            
            for alg_name, metrics in tracking_data.items():
                if metrics['execution_times']:
                    alg_names.append(alg_name)
                    avg_times.append(np.mean(metrics['execution_times']) * 1000)
            
            if alg_names:
                bars = ax.bar(alg_names, avg_times, color='skyblue', alpha=0.7)
                ax.set_title('跟踪算法执行时间对比')
                ax.set_ylabel('平均执行时间 (ms)')
                ax.tick_params(axis='x', rotation=45)
                
                # 添加数值标签
                for bar, time in zip(bars, avg_times):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{time:.2f}', ha='center', va='bottom')
        
        # 相似性算法对比
        if 'similarity' in results:
            ax = axes[0, 1]
            similarity_data = results['similarity']
            
            alg_names = []
            avg_times = []
            
            for alg_name, metrics in similarity_data.items():
                if metrics['execution_times']:
                    alg_names.append(alg_name)
                    avg_times.append(np.mean(metrics['execution_times']) * 1000)
            
            if alg_names:
                bars = ax.bar(alg_names, avg_times, color='lightgreen', alpha=0.7)
                ax.set_title('相似性算法执行时间对比')
                ax.set_ylabel('平均执行时间 (ms)')
                ax.tick_params(axis='x', rotation=45)
                
                for bar, time in zip(bars, avg_times):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{time:.2f}', ha='center', va='bottom')
        
        # 聚类算法轮廓系数对比
        if 'clustering' in results:
            ax = axes[1, 0]
            clustering_data = results['clustering']
            
            methods = []
            scores = []
            
            for method_name, result in clustering_data.items():
                if result.get('silhouette_score', -1) > -1:
                    methods.append(method_name.replace('_', '\n'))
                    scores.append(result['silhouette_score'])
            
            if methods:
                bars = ax.bar(methods, scores, color='orange', alpha=0.7)
                ax.set_title('聚类算法轮廓系数对比')
                ax.set_ylabel('轮廓系数')
                ax.tick_params(axis='x', rotation=45)
                
                for bar, score in zip(bars, scores):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
        
        # 综合性能雷达图
        ax = axes[1, 1]
        ax.text(0.5, 0.5, '综合性能评估\n(基于执行时间、精度、稳定性)', 
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"算法对比可视化结果已保存: {save_path}")

def run_algorithm_comparison(config, topographies, trajectories):
    """运行完整的算法对比"""
    benchmark = AlgorithmBenchmark(config)
    
    print("开始算法性能对比测试...")
    
    # 1. 跟踪算法对比
    tracking_results = benchmark.benchmark_tracking_algorithms(topographies)
    
    # 2. 相似性算法对比
    similarity_results = benchmark.benchmark_similarity_algorithms(trajectories)
    
    # 3. 聚类算法对比（需要特征数据）
    if trajectories:
        from src.trajectory_analysis import TrajectoryAnalyzer
        analyzer = TrajectoryAnalyzer(config)
        
        # 提取特征
        feature_data = {}
        for traj_id, traj_data in trajectories.items():
            features = analyzer.compute_trajectory_features(traj_data['trajectory'])
            if features:
                feature_data[traj_id] = features
        
        if feature_data:
            # 转换为特征矩阵
            feature_matrix = []
            for features in feature_data.values():
                feature_vector = [
                    features.get('total_distance', 0),
                    features.get('displacement', 0),
                    features.get('mean_velocity', 0),
                    features.get('tortuosity', 1),
                    features.get('straightness', 0),
                    features.get('complexity', 0)
                ]
                feature_matrix.append(feature_vector)
            
            feature_matrix = np.array(feature_matrix)
            
            # 标准化
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            feature_matrix = scaler.fit_transform(feature_matrix)
            
            clustering_results = benchmark.clustering_comparison.compare_clustering_methods(feature_matrix)
        else:
            clustering_results = {}
    else:
        clustering_results = {}
    
    # 生成报告
    results = {
        'tracking': tracking_results,
        'similarity': similarity_results,
        'clustering': clustering_results
    }
    
    report = benchmark.generate_comparison_report(
        tracking_results, similarity_results, clustering_results
    )
    
    # 保存报告
    import os
    report_path = os.path.join(config.RESULTS_ROOT, "algorithm_comparison_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 可视化结果
    viz_path = os.path.join(config.RESULTS_ROOT, "algorithm_comparison.png")
    benchmark.visualize_comparison_results(results, viz_path)
    
    print(f"算法对比报告已保存: {report_path}")
    print("\n" + "="*50)
    print("算法对比报告预览:")
    print("="*50)
    print(report[:1000] + "..." if len(report) > 1000 else report)
    
    return results, report