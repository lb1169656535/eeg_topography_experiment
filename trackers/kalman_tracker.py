import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict
from .base_tracker import BaseTracker

class KalmanTracker(BaseTracker):
    """卡尔曼预测跟踪器"""
    
    def __init__(self, config):
        super().__init__(config)
        self.algorithm_name = "kalman"
        self.description = "卡尔曼预测算法 - 基于运动预测"
        
        # 获取算法特定配置
        alg_config = config.get_algorithm_config('kalman')
        self.distance_threshold = alg_config.get('distance_threshold', 30.0)
        self.prediction_weight = alg_config.get('prediction_weight', 0.4)
        self.enable_reconnection = alg_config.get('enable_reconnection', True)
        self.max_inactive_frames = alg_config.get('max_inactive_frames', 30)
        
        self.algorithm_params = {
            'distance_threshold': self.distance_threshold,
            'prediction_weight': self.prediction_weight,
            'enable_reconnection': self.enable_reconnection,
            'max_inactive_frames': self.max_inactive_frames
        }
    
    def match_regions(self, current_regions: List[Dict], 
                     distance_threshold: float = None, frame_idx: int = 0) -> List[Tuple[int, int]]:
        """卡尔曼预测匹配算法"""
        if not current_regions:
            return []
        
        # 获取活跃区域
        active_regions = [r for r in self.regions if r.active]
        if not active_regions:
            return []
        
        # 使用配置的距离阈值
        if distance_threshold is None:
            distance_threshold = self.distance_threshold
        
        try:
            # 预测下一帧位置
            predicted_centers = []
            for region in active_regions:
                predicted_pos = self._predict_next_position(region)
                predicted_centers.append(predicted_pos)
            
            if not predicted_centers:
                return []
            
            predicted_centers = np.array(predicted_centers)
            current_centers = np.array([r['center'] for r in current_regions])
            
            # 计算距离矩阵
            distances = cdist(predicted_centers, current_centers)
            
            # 使用匈牙利算法求解最优分配
            try:
                row_indices, col_indices = linear_sum_assignment(distances)
                matches = []
                
                for row_idx, col_idx in zip(row_indices, col_indices):
                    if distances[row_idx, col_idx] < distance_threshold:
                        matches.append((row_idx, col_idx))
                        self.logger.debug(f"卡尔曼匹配轨迹{active_regions[row_idx].id}: "
                                        f"预测距离{distances[row_idx, col_idx]:.2f}")
                
                # 如果启用重连，尝试重连未匹配的区域
                if self.enable_reconnection and len(matches) < len(active_regions):
                    reconnection_matches = self._attempt_kalman_reconnection(
                        current_regions, active_regions, matches, frame_idx
                    )
                    matches.extend(reconnection_matches)
                
                self.logger.debug(f"卡尔曼算法第{frame_idx}帧: 匹配{len(matches)}对区域")
                return matches
                
            except Exception as e:
                self.logger.warning(f"卡尔曼匈牙利分配失败: {e}")
                return self._fallback_greedy_matching(current_regions, active_regions, distance_threshold)
            
        except Exception as e:
            self.logger.error(f"卡尔曼匹配失败: {e}")
            return []
    
    def _predict_next_position(self, region) -> np.ndarray:
        """预测下一个位置"""
        try:
            trajectory = region.trajectory
            
            if len(trajectory) < 2:
                # 如果轨迹点不足，返回当前位置
                return np.array(trajectory[-1])
            
            elif len(trajectory) == 2:
                # 简单线性预测
                velocity = np.array(trajectory[-1]) - np.array(trajectory[-2])
                predicted = np.array(trajectory[-1]) + velocity * self.prediction_weight
                
            else:
                # 使用多点预测，考虑加速度
                current_pos = np.array(trajectory[-1])
                prev_pos = np.array(trajectory[-2])
                prev_prev_pos = np.array(trajectory[-3])
                
                # 计算速度和加速度
                velocity = current_pos - prev_pos
                prev_velocity = prev_pos - prev_prev_pos
                acceleration = velocity - prev_velocity
                
                # 预测下一个位置（考虑速度和加速度）
                predicted = (current_pos + 
                           velocity * self.prediction_weight + 
                           acceleration * (self.prediction_weight ** 2) * 0.5)
            
            return predicted
            
        except Exception as e:
            self.logger.warning(f"位置预测失败: {e}")
            return np.array(region.trajectory[-1])
    
    def _attempt_kalman_reconnection(self, current_regions: List[Dict], 
                                   active_regions: List,
                                   existing_matches: List[Tuple[int, int]], 
                                   frame_idx: int) -> List[Tuple[int, int]]:
        """卡尔曼重连尝试"""
        reconnection_matches = []
        
        try:
            # 获取未匹配的区域索引
            matched_tracked = set([m[0] for m in existing_matches])
            matched_current = set([m[1] for m in existing_matches])
            
            unmatched_tracked = [i for i in range(len(active_regions)) if i not in matched_tracked]
            unmatched_current = [i for i in range(len(current_regions)) if i not in matched_current]
            
            if not unmatched_tracked or not unmatched_current:
                return reconnection_matches
            
            # 使用更宽松的阈值进行重连
            reconnection_threshold = self.distance_threshold * 1.8
            
            # 预测未匹配区域的位置
            for tracked_idx in unmatched_tracked:
                tracked_region = active_regions[tracked_idx]
                
                # 只为非活跃帧数不太多的区域尝试重连
                if tracked_region.inactive_frames >= self.max_inactive_frames:
                    continue
                
                predicted_pos = self._predict_next_position(tracked_region)
                
                best_match = -1
                best_distance = float('inf')
                
                for current_idx in unmatched_current:
                    current_region = current_regions[current_idx]
                    current_pos = np.array(current_region['center'])
                    
                    distance = np.linalg.norm(predicted_pos - current_pos)
                    
                    if distance < reconnection_threshold and distance < best_distance:
                        best_distance = distance
                        best_match = current_idx
                
                if best_match >= 0:
                    reconnection_matches.append((tracked_idx, best_match))
                    unmatched_current.remove(best_match)
                    
                    self.logger.debug(f"卡尔曼重连轨迹{tracked_region.id}: "
                                    f"预测距离{best_distance:.2f}")
            
        except Exception as e:
            self.logger.warning(f"卡尔曼重连失败: {e}")
        
        return reconnection_matches
    
    def _fallback_greedy_matching(self, current_regions: List[Dict], 
                                active_regions: List, 
                                distance_threshold: float) -> List[Tuple[int, int]]:
        """备用贪婪匹配"""
        try:
            current_centers = np.array([r['center'] for r in current_regions])
            tracked_centers = np.array([r.trajectory[-1] for r in active_regions])
            
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
            
        except Exception as e:
            self.logger.error(f"备用贪婪算法失败: {e}")
            return []
    
    def update_region_velocity_history(self, region, current_position: np.ndarray):
        """更新区域的速度历史"""
        try:
            if len(region.trajectory) >= 2:
                velocity = current_position - np.array(region.trajectory[-1])
                
                # 维护速度历史（最多保存10个历史速度）
                if not hasattr(region, 'velocity_history'):
                    region.velocity_history = []
                
                region.velocity_history.append(velocity)
                
                if len(region.velocity_history) > 10:
                    region.velocity_history.pop(0)
                    
        except Exception as e:
            self.logger.warning(f"速度历史更新失败: {e}")
    
    def get_algorithm_info(self) -> Dict:
        """获取算法信息"""
        info = super().get_algorithm_info()
        info.update({
            'description': self.description,
            'characteristics': [
                "运动预测",
                "考虑速度和加速度",
                "自适应阈值",
                "轨迹连续性优化"
            ],
            'advantages': [
                "对运动目标跟踪效果好",
                "能够处理短暂遮挡",
                "预测能力强",
                "轨迹平滑性好"
            ],
            'disadvantages': [
                "对非线性运动敏感",
                "需要足够的历史数据",
                "计算复杂度中等",
                "参数调优较复杂"
            ],
            'best_for': [
                "运动规律较明显的场景",
                "需要预测功能的场景",
                "轨迹连续性要求高的场景",
                "有遮挡的跟踪场景"
            ]
        })
        return info