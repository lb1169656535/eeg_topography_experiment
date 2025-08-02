import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict
from .base_tracker import BaseTracker

class HungarianTracker(BaseTracker):
    """匈牙利算法跟踪器 - 全局最优匹配"""
    
    def __init__(self, config):
        super().__init__(config)
        self.algorithm_name = "hungarian"
        self.description = "匈牙利算法 - 全局最优匹配解"
        
        # 获取算法特定配置
        alg_config = config.get_algorithm_config('hungarian')
        self.distance_threshold = alg_config.get('distance_threshold', 25.0)
        self.enable_reconnection = alg_config.get('enable_reconnection', True)
        self.max_inactive_frames = alg_config.get('max_inactive_frames', 25)
        
        # 匈牙利算法特定参数
        self.cost_threshold = self.distance_threshold * 2  # 成本阈值
        self.use_weighted_cost = True  # 是否使用加权成本
        
        self.algorithm_params = {
            'distance_threshold': self.distance_threshold,
            'cost_threshold': self.cost_threshold,
            'enable_reconnection': self.enable_reconnection,
            'max_inactive_frames': self.max_inactive_frames,
            'use_weighted_cost': self.use_weighted_cost
        }
    
    def match_regions(self, current_regions: List[Dict], 
                     distance_threshold: float = None, frame_idx: int = 0) -> List[Tuple[int, int]]:
        """匈牙利算法匹配实现"""
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
            # 构建成本矩阵
            cost_matrix = self._build_cost_matrix(current_regions, active_regions)
            
            if cost_matrix.size == 0:
                return []
            
            # 使用匈牙利算法求解最优分配
            try:
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
            except Exception as e:
                self.logger.warning(f"匈牙利算法求解失败: {e}, 使用备用贪婪算法")
                return self._fallback_greedy_matching(current_regions, active_regions, distance_threshold)
            
            # 筛选有效匹配
            matches = []
            for row_idx, col_idx in zip(row_indices, col_indices):
                cost = cost_matrix[row_idx, col_idx]
                
                # 只接受成本低于阈值的匹配
                if cost < self.cost_threshold:
                    matches.append((row_idx, col_idx))
                    self.logger.debug(f"匹配轨迹{active_regions[row_idx].id}与区域{col_idx}, 成本: {cost:.2f}")
            
            # 如果启用重连，尝试重连未匹配的区域
            if self.enable_reconnection and len(matches) < len(active_regions):
                reconnection_matches = self._attempt_hungarian_reconnection(
                    current_regions, active_regions, matches, frame_idx
                )
                matches.extend(reconnection_matches)
            
            self.logger.debug(f"匈牙利算法第{frame_idx}帧: 匹配{len(matches)}对区域")
            
            return matches
            
        except Exception as e:
            self.logger.error(f"匈牙利算法匹配失败: {e}")
            return self._fallback_greedy_matching(current_regions, active_regions, distance_threshold)
    
    def _build_cost_matrix(self, current_regions: List[Dict], active_regions: List) -> np.ndarray:
        """构建成本矩阵"""
        try:
            n_tracked = len(active_regions)
            n_current = len(current_regions)
            
            # 初始化成本矩阵
            cost_matrix = np.full((n_tracked, n_current), np.inf)
            
            # 计算各种成本组件
            for i, tracked_region in enumerate(active_regions):
                tracked_center = np.array(tracked_region.trajectory[-1])
                
                for j, current_region in enumerate(current_regions):
                    current_center = np.array(current_region['center'])
                    
                    # 计算综合成本
                    cost = self._calculate_assignment_cost(tracked_region, current_region, 
                                                         tracked_center, current_center)
                    cost_matrix[i, j] = cost
            
            return cost_matrix
            
        except Exception as e:
            self.logger.error(f"成本矩阵构建失败: {e}")
            return np.array([])
    
    def _calculate_assignment_cost(self, tracked_region, current_region, 
                                 tracked_center: np.ndarray, current_center: np.ndarray) -> float:
        """计算分配成本"""
        try:
            # 1. 基础距离成本
            distance = np.linalg.norm(tracked_center - current_center)
            distance_cost = distance
            
            # 如果距离超过阈值，返回极高成本
            if distance > self.distance_threshold:
                return np.inf
            
            if not self.use_weighted_cost:
                return distance_cost
            
            # 2. 强度差异成本
            intensity_diff = abs(tracked_region.intensity - current_region.get('intensity', 0))
            max_intensity = max(tracked_region.intensity, current_region.get('intensity', 0), 0.1)
            intensity_cost = intensity_diff / max_intensity * 10  # 权重为10
            
            # 3. 面积差异成本
            area_diff = abs(tracked_region.area - current_region.get('area', 0))
            max_area = max(tracked_region.area, current_region.get('area', 0), 1.0)
            area_cost = area_diff / max_area * 5  # 权重为5
            
            # 4. 速度一致性成本
            velocity_cost = 0
            if len(tracked_region.trajectory) >= 2:
                prev_velocity = np.array(tracked_region.trajectory[-1]) - np.array(tracked_region.trajectory[-2])
                current_velocity = current_center - tracked_center
                velocity_diff = np.linalg.norm(current_velocity - prev_velocity)
                velocity_cost = velocity_diff * 2  # 权重为2
            
            # 5. 非活跃帧数惩罚
            inactive_penalty = tracked_region.inactive_frames * 3  # 权重为3
            
            # 综合成本
            total_cost = (distance_cost + intensity_cost + area_cost + 
                         velocity_cost + inactive_penalty)
            
            return total_cost
            
        except Exception as e:
            self.logger.warning(f"成本计算失败: {e}")
            return distance if 'distance' in locals() else np.inf
    
    def _attempt_hungarian_reconnection(self, current_regions: List[Dict], 
                                      active_regions: List,
                                      existing_matches: List[Tuple[int, int]], 
                                      frame_idx: int) -> List[Tuple[int, int]]:
        """使用匈牙利算法尝试重连"""
        reconnection_matches = []
        
        try:
            # 获取未匹配的区域索引
            matched_tracked = set([m[0] for m in existing_matches])
            matched_current = set([m[1] for m in existing_matches])
            
            unmatched_tracked = [i for i in range(len(active_regions)) if i not in matched_tracked]
            unmatched_current = [i for i in range(len(current_regions)) if i not in matched_current]
            
            if not unmatched_tracked or not unmatched_current:
                return reconnection_matches
            
            # 构建重连成本矩阵（使用更宽松的阈值）
            reconnection_threshold = self.distance_threshold * 2.0
            cost_matrix = np.full((len(unmatched_tracked), len(unmatched_current)), np.inf)
            
            for i, tracked_idx in enumerate(unmatched_tracked):
                tracked_region = active_regions[tracked_idx]
                
                # 只为非活跃帧数不太多的区域尝试重连
                if tracked_region.inactive_frames >= self.max_inactive_frames:
                    continue
                
                tracked_center = np.array(tracked_region.trajectory[-1])
                
                for j, current_idx in enumerate(unmatched_current):
                    current_region = current_regions[current_idx]
                    current_center = np.array(current_region['center'])
                    
                    distance = np.linalg.norm(tracked_center - current_center)
                    
                    if distance < reconnection_threshold:
                        # 重连时的成本包含非活跃惩罚
                        reconnection_cost = distance + tracked_region.inactive_frames * 5
                        cost_matrix[i, j] = reconnection_cost
            
            # 如果有可重连的组合，使用匈牙利算法
            if np.any(cost_matrix < np.inf):
                try:
                    row_indices, col_indices = linear_sum_assignment(cost_matrix)
                    
                    for row_idx, col_idx in zip(row_indices, col_indices):
                        if cost_matrix[row_idx, col_idx] < np.inf:
                            original_tracked_idx = unmatched_tracked[row_idx]
                            original_current_idx = unmatched_current[col_idx]
                            
                            reconnection_matches.append((original_tracked_idx, original_current_idx))
                            
                            region = active_regions[original_tracked_idx]
                            self.logger.debug(f"重连轨迹{region.id}: 成本{cost_matrix[row_idx, col_idx]:.2f}")
                            
                except Exception as e:
                    self.logger.warning(f"重连匈牙利算法失败: {e}")
            
        except Exception as e:
            self.logger.warning(f"重连尝试失败: {e}")
        
        return reconnection_matches
    
    def _fallback_greedy_matching(self, current_regions: List[Dict], 
                                active_regions: List, 
                                distance_threshold: float) -> List[Tuple[int, int]]:
        """备用贪婪匹配算法"""
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
    
    def get_algorithm_info(self) -> Dict:
        """获取算法信息"""
        info = super().get_algorithm_info()
        info.update({
            'description': self.description,
            'characteristics': [
                "全局最优解",
                "成本矩阵优化",
                "多因素权衡",
                "高精度匹配"
            ],
            'advantages': [
                "全局最优匹配",
                "考虑多种特征",
                "匹配质量高",
                "数学基础扎实"
            ],
            'disadvantages': [
                "计算复杂度较高",
                "内存消耗较大",
                "参数调优复杂"
            ],
            'best_for': [
                "高精度要求场景",
                "轨迹质量优先场景",
                "复杂匹配问题",
                "离线分析场景"
            ]
        })
        return info