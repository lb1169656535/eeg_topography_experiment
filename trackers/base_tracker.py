import numpy as np
import cv2
from scipy.ndimage import label, center_of_mass
import logging
import time
from typing import List, Tuple, Dict, Optional
from abc import ABC, abstractmethod

class Region:
    """轨迹区域类"""
    def __init__(self, center: Tuple[float, float], area: float, intensity: float, id: int):
        self.center = center
        self.area = area
        self.intensity = intensity
        self.id = id
        self.trajectory = [center]
        self.active = True
        self.inactive_frames = 0
        self.max_inactive_frames = 25
        self.velocity_history = []
        self.predicted_position = None
        self.last_mask = None
        self.quality_score = 0.0

class BaseTracker(ABC):
    """基础跟踪器抽象类"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.regions = []
        self.next_region_id = 0
        
        # 算法名称（由子类设置）
        self.algorithm_name = "base"
        
        # 性能统计
        self.performance_stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_matches': 0,
            'computation_times': [],
            'memory_usage': []
        }
    
    def detect_high_activation_regions(self, topography: np.ndarray, frame_idx: int = 0) -> List[Dict]:
        """检测高激活区域 - 通用实现"""
        try:
            # 只考虑非零区域（头部内部）
            valid_mask = topography != 0
            if not np.any(valid_mask):
                return []
            
            valid_values = topography[valid_mask]
            
            # 自适应阈值计算
            threshold = np.percentile(valid_values, self.config.THRESHOLD_PERCENTILE)
            
            # 二值化
            binary = (topography > threshold) & valid_mask
            
            if not np.any(binary):
                # 降低阈值重试
                threshold = np.percentile(valid_values, max(70, self.config.THRESHOLD_PERCENTILE - 15))
                binary = (topography > threshold) & valid_mask
                
                if not np.any(binary):
                    return []
            
            # 形态学操作清理噪声
            try:
                kernel = np.ones((3, 3), np.uint8)
                binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            except:
                pass
            
            # 连通域分析
            labeled_array, num_features = label(binary)
            
            regions = []
            for i in range(1, num_features + 1):
                region_mask = labeled_array == i
                area = np.sum(region_mask)
                
                if area < self.config.MIN_REGION_SIZE:
                    continue
                
                # 计算质心
                center = center_of_mass(region_mask)
                
                # 计算强度统计
                region_values = topography[region_mask]
                intensity = np.mean(region_values)
                max_intensity = np.max(region_values)
                
                regions.append({
                    'center': center,
                    'area': area,
                    'intensity': intensity,
                    'max_intensity': max_intensity,
                    'mask': region_mask,
                    'threshold_used': threshold
                })
            
            # 按强度和面积的组合排序
            regions.sort(key=lambda x: x['intensity'] * np.sqrt(x['area']), reverse=True)
            selected_regions = regions[:self.config.MAX_REGIONS]
            
            return selected_regions
            
        except Exception as e:
            self.logger.error(f"区域检测失败: {e}")
            return []
    
    @abstractmethod
    def match_regions(self, current_regions: List[Dict], 
                     distance_threshold: float = 20.0, frame_idx: int = 0) -> List[Tuple[int, int]]:
        """匹配区域 - 由子类实现具体算法"""
        pass
    
    def update_tracker(self, topography: np.ndarray, frame_idx: int = 0) -> Dict:
        """更新跟踪器 - 通用框架"""
        start_time = time.time()
        
        try:
            # 检测当前帧的区域
            current_regions = self.detect_high_activation_regions(topography, frame_idx)
            
            # 使用具体算法进行匹配
            matches = self.match_regions(current_regions, frame_idx=frame_idx)
            
            # 更新匹配的区域
            active_regions = [r for r in self.regions if r.active]
            matched_tracked = set()
            matched_current = set()
            
            for tracked_idx, current_idx in matches:
                if tracked_idx < len(active_regions) and current_idx < len(current_regions):
                    region = active_regions[tracked_idx]
                    current_region = current_regions[current_idx]
                    
                    # 更新轨迹
                    region.trajectory.append(current_region['center'])
                    region.area = current_region['area']
                    region.intensity = current_region['intensity']
                    region.inactive_frames = 0
                    region.last_mask = current_region.get('mask')
                    
                    matched_tracked.add(tracked_idx)
                    matched_current.add(current_idx)
            
            # 处理未匹配的跟踪区域
            for i, region in enumerate(active_regions):
                if i not in matched_tracked:
                    region.inactive_frames += 1
                    
                    if region.inactive_frames >= region.max_inactive_frames:
                        region.active = False
            
            # 为未匹配的当前区域创建新的跟踪区域
            for i, current_region in enumerate(current_regions):
                if i not in matched_current:
                    new_region = Region(
                        center=current_region['center'],
                        area=current_region['area'],
                        intensity=current_region['intensity'],
                        id=self.next_region_id
                    )
                    new_region.last_mask = current_region.get('mask')
                    self.regions.append(new_region)
                    self.next_region_id += 1
            
            # 更新性能统计
            computation_time = time.time() - start_time
            self.performance_stats['computation_times'].append(computation_time)
            self.performance_stats['total_frames'] += 1
            self.performance_stats['total_detections'] += len(current_regions)
            self.performance_stats['total_matches'] += len(matches)
            
            return {
                'current_regions': current_regions,
                'tracked_regions': [r for r in self.regions if r.active],
                'all_regions': self.regions,
                'frame_idx': frame_idx,
                'matches': matches,
                'algorithm': self.algorithm_name
            }
            
        except Exception as e:
            self.logger.error(f"跟踪更新失败: {e}")
            return {
                'current_regions': [],
                'tracked_regions': [],
                'all_regions': self.regions,
                'frame_idx': frame_idx,
                'error': str(e),
                'algorithm': self.algorithm_name
            }
    
    def track_sequence(self, topographies: np.ndarray) -> Dict:
        """跟踪整个序列"""
        n_frames = topographies.shape[0]
        tracking_results = []
        
        # 重置跟踪器
        self.reset_tracker()
        
        self.logger.info(f"开始使用{self.algorithm_name}算法跟踪{n_frames}帧")
        
        start_time = time.time()
        
        try:
            for frame_idx in range(n_frames):
                topography = topographies[frame_idx]
                result = self.update_tracker(topography, frame_idx)
                result['frame'] = frame_idx
                tracking_results.append(result)
            
            total_time = time.time() - start_time
            
            # 提取轨迹
            trajectories = self.extract_trajectories()
            
            # 计算性能指标
            metrics = self.calculate_performance_metrics(trajectories, total_time)
            
            self.logger.info(f"{self.algorithm_name}算法完成: {len(trajectories)}条轨迹, "
                           f"耗时{total_time:.2f}秒")
            
            return {
                'algorithm': self.algorithm_name,
                'frame_results': tracking_results,
                'trajectories': trajectories,
                'metrics': metrics,
                'summary': {
                    'total_regions': len(self.regions),
                    'tracked_regions': len(trajectories),
                    'total_frames': n_frames,
                    'total_time': total_time
                }
            }
            
        except Exception as e:
            self.logger.error(f"{self.algorithm_name}算法跟踪失败: {e}")
            return {
                'algorithm': self.algorithm_name,
                'frame_results': tracking_results,
                'trajectories': {},
                'metrics': {},
                'summary': {
                    'total_regions': 0,
                    'tracked_regions': 0,
                    'total_frames': n_frames,
                    'error': str(e)
                }
            }
    
    def reset_tracker(self):
        """重置跟踪器状态"""
        self.regions = []
        self.next_region_id = 0
        self.performance_stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_matches': 0,
            'computation_times': [],
            'memory_usage': []
        }
    
    def extract_trajectories(self) -> Dict:
        """提取有效轨迹"""
        trajectories = {}
        
        for region in self.regions:
            if len(region.trajectory) > 2:  # 至少跟踪了3帧
                try:
                    trajectory_array = np.array(region.trajectory)
                    
                    # 计算基本统计
                    distances = np.linalg.norm(np.diff(trajectory_array, axis=0), axis=1)
                    total_distance = np.sum(distances)
                    avg_velocity = np.mean(distances) if len(distances) > 0 else 0
                    
                    # 计算轨迹质量分数
                    quality_score = self.compute_trajectory_quality(region)
                    
                    trajectories[region.id] = {
                        'trajectory': trajectory_array,
                        'length': len(region.trajectory),
                        'mean_intensity': float(getattr(region, 'intensity', 0.0)),
                        'area': float(getattr(region, 'area', 0.0)),
                        'total_distance': float(total_distance),
                        'avg_velocity': float(avg_velocity),
                        'inactive_frames': getattr(region, 'inactive_frames', 0),
                        'quality_score': float(quality_score)
                    }
                except Exception as e:
                    self.logger.warning(f"提取轨迹{region.id}失败: {e}")
                    continue
        
        return trajectories
    
    def compute_trajectory_quality(self, region: Region) -> float:
        """计算轨迹质量分数"""
        try:
            trajectory = np.array(region.trajectory)
            
            if len(trajectory) < 2:
                return 0.0
            
            # 长度分数
            length_score = min(1.0, len(trajectory) / 50.0)
            
            # 连续性分数
            continuity_score = max(0.0, 1.0 - region.inactive_frames / region.max_inactive_frames)
            
            # 运动平滑性分数
            if len(trajectory) >= 3:
                velocities = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
                if len(velocities) > 1:
                    velocity_var = np.var(velocities)
                    smoothness_score = max(0.0, 1.0 - velocity_var / 50.0)
                else:
                    smoothness_score = 0.8
            else:
                smoothness_score = 0.5
            
            # 强度一致性分数
            intensity_score = min(1.0, getattr(region, 'intensity', 0) / 0.5)
            
            # 综合质量分数
            quality_score = (length_score * 0.3 + 
                           continuity_score * 0.3 + 
                           smoothness_score * 0.25 + 
                           intensity_score * 0.15)
            
            return quality_score
            
        except Exception as e:
            self.logger.warning(f"质量计算失败: {e}")
            return 0.0
    
    def calculate_performance_metrics(self, trajectories: Dict, total_time: float) -> Dict:
        """计算性能指标"""
        metrics = {}
        
        try:
            # 基本指标
            metrics['trajectory_count'] = len(trajectories)
            metrics['computation_time'] = total_time
            
            if trajectories:
                lengths = [traj['length'] for traj in trajectories.values()]
                qualities = [traj['quality_score'] for traj in trajectories.values()]
                
                metrics['average_trajectory_length'] = np.mean(lengths)
                metrics['max_trajectory_length'] = np.max(lengths)
                metrics['trajectory_quality'] = np.mean(qualities)
                
                # 连续性指标
                metrics['tracking_continuity'] = self._calculate_continuity(trajectories)
                metrics['trajectory_smoothness'] = self._calculate_smoothness(trajectories)
            else:
                metrics['average_trajectory_length'] = 0
                metrics['max_trajectory_length'] = 0
                metrics['trajectory_quality'] = 0
                metrics['tracking_continuity'] = 0
                metrics['trajectory_smoothness'] = 0
            
            # 检测稳定性
            if self.performance_stats['computation_times']:
                avg_time = np.mean(self.performance_stats['computation_times'])
                time_std = np.std(self.performance_stats['computation_times'])
                metrics['detection_stability'] = 1.0 / (1.0 + time_std / avg_time) if avg_time > 0 else 0
            else:
                metrics['detection_stability'] = 0
            
            # 内存使用（简化版）
            try:
                import psutil
                metrics['memory_usage'] = psutil.Process().memory_info().rss / 1024 / 1024
            except ImportError:
                metrics['memory_usage'] = 0
            
        except Exception as e:
            self.logger.error(f"性能指标计算失败: {e}")
            for metric in self.config.EVALUATION_METRICS:
                metrics[metric] = 0.0
        
        return metrics
    
    def _calculate_continuity(self, trajectories: Dict) -> float:
        """计算跟踪连续性"""
        if not trajectories:
            return 0.0
        
        continuity_scores = []
        for traj_data in trajectories.values():
            trajectory = traj_data['trajectory']
            if len(trajectory) > 2:
                velocities = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
                if len(velocities) > 1:
                    velocity_stability = 1.0 / (1.0 + np.std(velocities))
                    continuity_scores.append(velocity_stability)
        
        return np.mean(continuity_scores) if continuity_scores else 0.0
    
    def _calculate_smoothness(self, trajectories: Dict) -> float:
        """计算轨迹平滑度"""
        if not trajectories:
            return 0.0
        
        smoothness_scores = []
        for traj_data in trajectories.values():
            trajectory = traj_data['trajectory']
            if len(trajectory) > 3:
                velocities = np.diff(trajectory, axis=0)
                accelerations = np.diff(velocities, axis=0)
                if len(accelerations) > 0:
                    acceleration_magnitude = np.linalg.norm(accelerations, axis=1)
                    if len(acceleration_magnitude) > 1:
                        smoothness = 1.0 / (1.0 + np.std(acceleration_magnitude))
                        smoothness_scores.append(smoothness)
        
        return np.mean(smoothness_scores) if smoothness_scores else 0.0
    
    def get_algorithm_info(self) -> Dict:
        """获取算法信息"""
        return {
            'name': self.algorithm_name,
            'description': getattr(self, 'description', '基础跟踪算法'),
            'parameters': getattr(self, 'algorithm_params', {}),
            'performance_stats': self.performance_stats
        }