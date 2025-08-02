import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, euclidean
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging

# 使用fastdtw替代dtw
try:
    from fastdtw import fastdtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False

class TrajectoryAnalyzer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if not DTW_AVAILABLE:
            self.logger.warning("FastDTW not available, using Euclidean distance for trajectory comparison")
    
    def compute_trajectory_features(self, trajectory: np.ndarray) -> Dict:
        """计算轨迹特征"""
        if len(trajectory) < 2:
            return {}
        
        try:
            # 基本几何特征
            length = len(trajectory)
            
            # 计算逐步距离
            step_distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
            total_distance = np.sum(step_distances)
            displacement = np.linalg.norm(trajectory[-1] - trajectory[0])
            
            # 速度特征
            mean_velocity = np.mean(step_distances) if len(step_distances) > 0 else 0
            max_velocity = np.max(step_distances) if len(step_distances) > 0 else 0
            velocity_std = np.std(step_distances) if len(step_distances) > 0 else 0
            
            # 方向特征
            if length > 2:
                directions = np.diff(trajectory, axis=0)
                angles = np.arctan2(directions[:, 1], directions[:, 0])
                angle_changes = np.diff(angles)
                # 处理角度跳跃
                angle_changes = np.mod(angle_changes + np.pi, 2*np.pi) - np.pi
                mean_angle_change = np.mean(np.abs(angle_changes))
                tortuosity = total_distance / (displacement + 1e-8)
            else:
                mean_angle_change = 0
                tortuosity = 1
            
            # 覆盖区域
            if length > 2:
                min_coords = np.min(trajectory, axis=0)
                max_coords = np.max(trajectory, axis=0)
                ranges = max_coords - min_coords
                bounding_area = np.prod(ranges) if np.all(ranges > 0) else 0
            else:
                bounding_area = 0
            
            # 质心和散布
            centroid = np.mean(trajectory, axis=0)
            distances_to_centroid = np.linalg.norm(trajectory - centroid, axis=1)
            mean_spread = np.mean(distances_to_centroid)
            max_spread = np.max(distances_to_centroid)
            
            # 轨迹复杂度
            complexity = np.sum(np.abs(np.diff(step_distances))) if len(step_distances) > 1 else 0
            
            return {
                'length': length,
                'total_distance': total_distance,
                'displacement': displacement,
                'mean_velocity': mean_velocity,
                'max_velocity': max_velocity,
                'velocity_std': velocity_std,
                'mean_angle_change': mean_angle_change,
                'tortuosity': tortuosity,
                'bounding_area': bounding_area,
                'straightness': displacement / (total_distance + 1e-8),
                'mean_spread': mean_spread,
                'max_spread': max_spread,
                'complexity': complexity
            }
            
        except Exception as e:
            self.logger.error(f"Error computing trajectory features: {e}")
            return {}
    
    def compute_dtw_distance(self, traj1: np.ndarray, traj2: np.ndarray) -> float:
        """计算两条轨迹之间的DTW距离"""
        try:
            if DTW_AVAILABLE:
                distance, _ = fastdtw(traj1, traj2, dist=euclidean)
                return distance
            else:
                # 使用形状匹配的欧几里得距离作为后备
                return self.compute_shape_distance(traj1, traj2)
        except Exception as e:
            self.logger.warning(f"DTW computation failed: {e}, using shape distance")
            return self.compute_shape_distance(traj1, traj2)
    
    def compute_shape_distance(self, traj1: np.ndarray, traj2: np.ndarray) -> float:
        """计算轨迹形状距离（当DTW不可用时的后备方案）"""
        try:
            # 标准化轨迹长度
            from scipy.interpolate import interp1d
            
            # 选择较短轨迹的长度作为标准长度
            target_length = min(len(traj1), len(traj2), 50)  # 限制最大长度以提高效率
            
            if len(traj1) < 2 or len(traj2) < 2:
                return np.linalg.norm(traj1.flatten() - traj2.flatten())
            
            # 创建插值函数
            t1 = np.linspace(0, 1, len(traj1))
            t2 = np.linspace(0, 1, len(traj2))
            t_new = np.linspace(0, 1, target_length)
            
            # 对每个维度进行插值
            traj1_interp = np.zeros((target_length, traj1.shape[1]))
            traj2_interp = np.zeros((target_length, traj2.shape[1]))
            
            for dim in range(traj1.shape[1]):
                f1 = interp1d(t1, traj1[:, dim], kind='linear', bounds_error=False, fill_value='extrapolate')
                f2 = interp1d(t2, traj2[:, dim], kind='linear', bounds_error=False, fill_value='extrapolate')
                traj1_interp[:, dim] = f1(t_new)
                traj2_interp[:, dim] = f2(t_new)
            
            # 计算欧几里得距离
            return np.linalg.norm(traj1_interp - traj2_interp)
            
        except Exception as e:
            self.logger.warning(f"Shape distance computation failed: {e}")
            # 最后的后备方案：简单的端点距离
            return euclidean(traj1[-1], traj2[-1]) + euclidean(traj1[0], traj2[0])
    
    def compute_trajectory_similarity_matrix(self, trajectories: Dict) -> np.ndarray:
        """计算轨迹相似性矩阵"""
        trajectory_list = list(trajectories.values())
        n_trajectories = len(trajectory_list)
        
        if n_trajectories < 2:
            return np.array([[1.0]])
        
        # 计算距离矩阵
        distance_matrix = np.zeros((n_trajectories, n_trajectories))
        
        self.logger.info(f"Computing similarity matrix for {n_trajectories} trajectories")
        
        for i in range(n_trajectories):
            for j in range(i+1, n_trajectories):
                traj1 = trajectory_list[i]['trajectory']
                traj2 = trajectory_list[j]['trajectory']
                
                try:
                    distance = self.compute_dtw_distance(traj1, traj2)
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
                except Exception as e:
                    self.logger.warning(f"Distance computation failed for trajectories {i}, {j}: {e}")
                    distance_matrix[i, j] = float('inf')
                    distance_matrix[j, i] = float('inf')
        
        # 转换为相似性矩阵
        max_distance = np.max(distance_matrix[distance_matrix != float('inf')])
        if max_distance > 0:
            # 处理无穷大值
            distance_matrix[distance_matrix == float('inf')] = max_distance * 2
            similarity_matrix = 1 - distance_matrix / (max_distance * 2)
        else:
            similarity_matrix = np.ones_like(distance_matrix)
        
        # 确保对角线为1
        np.fill_diagonal(similarity_matrix, 1.0)
        
        return similarity_matrix
    
    def cluster_trajectories(self, trajectories: Dict, method: str = 'hierarchical',
                           n_clusters: Optional[int] = None) -> Dict:
        """聚类轨迹"""
        if len(trajectories) < 2:
            return {'labels': [0], 'n_clusters': 1, 'trajectory_ids': list(trajectories.keys())}
        
        # 计算特征矩阵
        features = []
        trajectory_ids = []
        
        for traj_id, traj_data in trajectories.items():
            traj_features = self.compute_trajectory_features(traj_data['trajectory'])
            if traj_features:  # 确保特征不为空
                feature_vector = [
                    traj_features.get('total_distance', 0),
                    traj_features.get('displacement', 0),
                    traj_features.get('mean_velocity', 0),
                    traj_features.get('tortuosity', 1),
                    traj_features.get('straightness', 0),
                    traj_features.get('bounding_area', 0),
                    traj_features.get('mean_spread', 0),
                    traj_features.get('complexity', 0)
                ]
                features.append(feature_vector)
                trajectory_ids.append(traj_id)
        
        if len(features) < 2:
            return {'labels': [0], 'n_clusters': 1, 'trajectory_ids': trajectory_ids}
        
        features = np.array(features)
        
        # 标准化特征
        try:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
        except Exception as e:
            self.logger.warning(f"Feature scaling failed: {e}, using original features")
            features_scaled = features
        
        # 处理NaN和无穷大值
        features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
        
        try:
            if method == 'hierarchical':
                labels, n_clusters = self._hierarchical_clustering(features_scaled, n_clusters)
            elif method == 'kmeans':
                labels, n_clusters = self._kmeans_clustering(features_scaled, n_clusters)
            elif method == 'dbscan':
                labels, n_clusters = self._dbscan_clustering(features_scaled)
            else:
                self.logger.error(f"Unknown clustering method: {method}")
                return {'labels': [0] * len(features), 'n_clusters': 1, 'trajectory_ids': trajectory_ids}
            
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            return {'labels': [0] * len(features), 'n_clusters': 1, 'trajectory_ids': trajectory_ids}
        
        return {
            'labels': labels,
            'n_clusters': n_clusters,
            'trajectory_ids': trajectory_ids,
            'features': features_scaled
        }
    
    def _hierarchical_clustering(self, features: np.ndarray, n_clusters: Optional[int]) -> Tuple[np.ndarray, int]:
        """层次聚类"""
        linkage_matrix = linkage(features, method='ward')
        
        if n_clusters is None:
            # 自动确定聚类数
            max_clusters = min(len(features) // 2, 5)
            best_score = -1
            best_n_clusters = 2
            
            for n in range(2, max_clusters + 1):
                try:
                    labels = fcluster(linkage_matrix, n, criterion='maxclust')
                    if len(np.unique(labels)) > 1:
                        score = silhouette_score(features, labels)
                        if score > best_score:
                            best_score = score
                            best_n_clusters = n
                except Exception as e:
                    self.logger.warning(f"Silhouette score computation failed for n={n}: {e}")
                    continue
            
            n_clusters = best_n_clusters
        
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        return labels, n_clusters
    
    def _kmeans_clustering(self, features: np.ndarray, n_clusters: Optional[int]) -> Tuple[np.ndarray, int]:
        """K-means聚类"""
        if n_clusters is None:
            n_clusters = min(len(features) // 2, 3)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        return labels, n_clusters
    
    def _dbscan_clustering(self, features: np.ndarray) -> Tuple[np.ndarray, int]:
        """DBSCAN聚类"""
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        labels = dbscan.fit_predict(features)
        n_clusters = len(np.unique(labels[labels >= 0]))
        return labels, n_clusters
    
    def analyze_subject_consistency(self, subject_trajectories: Dict) -> Dict:
        """分析单个被试的轨迹一致性"""
        results = {}
        
        for subject_id, sessions in subject_trajectories.items():
            self.logger.info(f"Analyzing consistency for subject {subject_id}")
            subject_results = {}
            
            # 收集该被试的所有轨迹
            all_trajectories = {}
            for session_id, trajectories in sessions.items():
                for traj_id, traj_data in trajectories.items():
                    key = f"{session_id}_{traj_id}"
                    all_trajectories[key] = traj_data
            
            if len(all_trajectories) > 1:
                try:
                    # 计算相似性矩阵
                    similarity_matrix = self.compute_trajectory_similarity_matrix(all_trajectories)
                    
                    # 聚类分析
                    clustering_results = self.cluster_trajectories(all_trajectories)
                    
                    # 计算一致性指标
                    upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
                    mean_similarity = np.mean(upper_triangle)
                    consistency_score = mean_similarity
                    
                    subject_results = {
                        'n_trajectories': len(all_trajectories),
                        'mean_similarity': float(mean_similarity),
                        'consistency_score': float(consistency_score),
                        'n_clusters': clustering_results['n_clusters'],
                        'clustering': clustering_results,
                        'similarity_matrix': similarity_matrix
                    }
                    
                except Exception as e:
                    self.logger.error(f"Consistency analysis failed for subject {subject_id}: {e}")
                    subject_results = {
                        'n_trajectories': len(all_trajectories),
                        'error': str(e)
                    }
            else:
                subject_results = {
                    'n_trajectories': len(all_trajectories),
                    'note': 'Insufficient trajectories for analysis'
                }
            
            results[subject_id] = subject_results
        
        return results
    
    def compare_subjects(self, subject_trajectories: Dict) -> Dict:
        """比较不同被试之间的差异"""
        # 计算每个被试的总体特征
        subject_features = {}
        
        for subject_id, sessions in subject_trajectories.items():
            all_trajectories = {}
            for session_id, trajectories in sessions.items():
                for traj_id, traj_data in trajectories.items():
                    key = f"{session_id}_{traj_id}"
                    all_trajectories[key] = traj_data
            
            if all_trajectories:
                # 计算平均特征
                features_list = []
                for traj_data in all_trajectories.values():
                    traj_features = self.compute_trajectory_features(traj_data['trajectory'])
                    if traj_features:
                        features_list.append(traj_features)
                
                if features_list:
                    # 计算平均特征
                    mean_features = {}
                    for key in features_list[0].keys():
                        values = [f[key] for f in features_list if key in f and not np.isnan(f[key])]
                        mean_features[key] = np.mean(values) if values else 0
                    
                    subject_features[subject_id] = mean_features
        
        # 计算被试间距离矩阵
        inter_subject_similarity = {}
        if len(subject_features) > 1:
            subject_ids = list(subject_features.keys())
            n_subjects = len(subject_ids)
            distance_matrix = np.zeros((n_subjects, n_subjects))
            
            # 提取特征向量
            feature_keys = list(subject_features[subject_ids[0]].keys())
            subject_vectors = []
            
            for subject_id in subject_ids:
                vector = [subject_features[subject_id][key] for key in feature_keys]
                subject_vectors.append(vector)
            
            subject_vectors = np.array(subject_vectors)
            
            # 计算距离矩阵
            for i in range(n_subjects):
                for j in range(i+1, n_subjects):
                    dist = euclidean(subject_vectors[i], subject_vectors[j])
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
            
            # 转换为相似性
            max_dist = np.max(distance_matrix)
            if max_dist > 0:
                similarity_matrix = 1 - distance_matrix / max_dist
            else:
                similarity_matrix = np.ones_like(distance_matrix)
            
            np.fill_diagonal(similarity_matrix, 1.0)
            inter_subject_similarity = {
                'similarity_matrix': similarity_matrix,
                'subject_ids': subject_ids
            }
        
        return {
            'subject_features': subject_features,
            'n_subjects': len(subject_features),
            'inter_subject_similarity': inter_subject_similarity
        }
    
    def generate_summary_report(self, analysis_results: Dict) -> str:
        """生成分析报告"""
        report = []
        report.append("=" * 60)
        report.append("EEG轨迹分析报告")
        report.append("=" * 60)
        report.append("")
        
        # 总体统计
        if 'subject_consistency' in analysis_results:
            consistency_results = analysis_results['subject_consistency']
            n_subjects = len(consistency_results)
            report.append(f"分析被试数量: {n_subjects}")
            
            # 一致性统计
            consistency_scores = []
            trajectory_counts = []
            
            for subject_id, results in consistency_results.items():
                if 'consistency_score' in results:
                    consistency_scores.append(results['consistency_score'])
                    trajectory_counts.append(results['n_trajectories'])
            
            if consistency_scores:
                mean_consistency = np.mean(consistency_scores)
                std_consistency = np.std(consistency_scores)
                min_consistency = np.min(consistency_scores)
                max_consistency = np.max(consistency_scores)
                
                report.append(f"平均一致性得分: {mean_consistency:.3f} ± {std_consistency:.3f}")
                report.append(f"一致性得分范围: {min_consistency:.3f} - {max_consistency:.3f}")
                
                total_trajectories = sum(trajectory_counts)
                mean_trajectories = np.mean(trajectory_counts)
                report.append(f"总轨迹数量: {total_trajectories}")
                report.append(f"平均每被试轨迹数: {mean_trajectories:.1f}")
            
            report.append("")
        
        # 各被试详细结果
        if 'subject_consistency' in analysis_results:
            report.append("各被试一致性分析详情:")
            report.append("-" * 40)
            
            for subject_id, results in consistency_results.items():
                report.append(f"被试 {subject_id}:")
                
                if 'error' in results:
                    report.append(f"  分析出错: {results['error']}")
                elif 'note' in results:
                    report.append(f"  {results['note']}")
                    report.append(f"  轨迹数量: {results['n_trajectories']}")
                else:
                    report.append(f"  轨迹数量: {results['n_trajectories']}")
                    if 'consistency_score' in results:
                        report.append(f"  一致性得分: {results['consistency_score']:.3f}")
                        report.append(f"  聚类数量: {results['n_clusters']}")
                        
                        # 如果有聚类信息，显示聚类分布
                        if 'clustering' in results and 'labels' in results['clustering']:
                            labels = results['clustering']['labels']
                            unique_labels, counts = np.unique(labels, return_counts=True)
                            cluster_info = [f"簇{label}: {count}条轨迹" for label, count in zip(unique_labels, counts)]
                            report.append(f"  聚类分布: {', '.join(cluster_info)}")
                
                report.append("")
        
        # 被试间比较
        if 'subject_comparison' in analysis_results:
            comparison_results = analysis_results['subject_comparison']
            report.append("被试间比较:")
            report.append("-" * 40)
            
            if 'inter_subject_similarity' in comparison_results:
                similarity_info = comparison_results['inter_subject_similarity']
                if 'similarity_matrix' in similarity_info:
                    similarity_matrix = similarity_info['similarity_matrix']
                    # 计算平均被试间相似性（排除对角线）
                    upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
                    mean_inter_similarity = np.mean(upper_triangle)
                    report.append(f"平均被试间相似性: {mean_inter_similarity:.3f}")
                    
                    # 找出最相似和最不相似的被试对
                    subject_ids = similarity_info['subject_ids']
                    max_idx = np.unravel_index(np.argmax(upper_triangle), 
                                             (len(subject_ids), len(subject_ids)))
                    min_idx = np.unravel_index(np.argmin(upper_triangle), 
                                             (len(subject_ids), len(subject_ids)))
                    
                    if len(subject_ids) > 1:
                        # 重新计算上三角矩阵的索引
                        triu_indices = np.triu_indices_from(similarity_matrix, k=1)
                        max_pos = np.argmax(upper_triangle)
                        min_pos = np.argmin(upper_triangle)
                        
                        max_i, max_j = triu_indices[0][max_pos], triu_indices[1][max_pos]
                        min_i, min_j = triu_indices[0][min_pos], triu_indices[1][min_pos]
                        
                        report.append(f"最相似被试对: {subject_ids[max_i]} - {subject_ids[max_j]} "
                                    f"(相似性: {similarity_matrix[max_i, max_j]:.3f})")
                        report.append(f"最不相似被试对: {subject_ids[min_i]} - {subject_ids[min_j]} "
                                    f"(相似性: {similarity_matrix[min_i, min_j]:.3f})")
            
            report.append("")
        
        # 添加方法说明
        report.append("分析方法说明:")
        report.append("-" * 40)
        report.append("• 轨迹特征: 包括总距离、位移、速度、弯曲度、直线度等")
        if DTW_AVAILABLE:
            report.append("• 相似性计算: 使用动态时间规整(DTW)算法")
        else:
            report.append("• 相似性计算: 使用形状匹配的欧几里得距离")
        report.append("• 聚类方法: 层次聚类，自动确定最佳聚类数")
        report.append("• 一致性评分: 基于轨迹间平均相似性计算")
        
        return "\n".join(report)