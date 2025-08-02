import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict
from .base_tracker import BaseTracker

class HungarianTracker(BaseTracker):
    """匈牙利算法跟踪器 - 全局最优匹配（增强版）"""
    
    def __init__(self, config):
        super().__init__(config)
        self.algorithm_name = "hungarian"
        self.description = "匈牙利算法 - 全局最优匹配解（增强版）"
        
        # 获取算法特定配置
        alg_config = config.get_algorithm_config('hungarian')
        self.distance_threshold = alg_config.get('distance_threshold', 25.0)
        self.enable_reconnection = alg_config.get('enable_reconnection', True)
        self.max_inactive_frames = alg_config.get('max_inactive_frames', 25)
        
        # 匈牙利算法特定参数 - 优化后的参数
        self.cost_threshold = self.distance_threshold * 3  # 增加成本阈值
        self.use_weighted_cost = True
        self.adaptive_threshold = True  # 新增：自适应阈值
        self.fallback_enabled = True   # 新增：启用备用算法
        
        self.algorithm_params = {
            'distance_threshold': self.distance_threshold,
            'cost_threshold': self.cost_threshold,
            'enable_reconnection': self.enable_reconnection,
            'max_inactive_frames': self.max_inactive_frames,
            'use_weighted_cost': self.use_weighted_cost,
            'adaptive_threshold': self.adaptive_threshold
        }
    
    def match_regions(self, current_regions: List[Dict], 
                     distance_threshold: float = None, frame_idx: int = 0) -> List[Tuple[int, int]]:
        """匈牙利算法匹配实现（增强版）"""
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
            cost_matrix = self._build_robust_cost_matrix(current_regions, active_regions, distance_threshold)
            
            if cost_matrix.size == 0:
                self.logger.warning("成本矩阵为空，使用备用算法")
                return self._fallback_greedy_matching(current_regions, active_regions, distance_threshold)
            
            # 检查成本矩阵的可行性
            if not self._is_matrix_feasible(cost_matrix):
                self.logger.warning(f"成本矩阵不可行（帧{frame_idx}），尝试自适应调整")
                
                # 尝试自适应调整
                if self.adaptive_threshold:
                    adjusted_matrix = self._adjust_cost_matrix(cost_matrix, distance_threshold)
                    if self._is_matrix_feasible(adjusted_matrix):
                        cost_matrix = adjusted_matrix
                        self.logger.debug("成本矩阵自适应调整成功")
                    else:
                        self.logger.warning("自适应调整失败，使用备用贪婪算法")
                        return self._fallback_greedy_matching(current_regions, active_regions, distance_threshold)
                else:
                    self.logger.warning("成本矩阵不可行，使用备用贪婪算法")
                    return self._fallback_greedy_matching(current_regions, active_regions, distance_threshold)
            
            # 使用匈牙利算法求解最优分配
            try:
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                
                # 筛选有效匹配
                matches = []
                for row_idx, col_idx in zip(row_indices, col_indices):
                    cost = cost_matrix[row_idx, col_idx]
                    
                    # 只接受成本低于阈值的匹配
                    if cost < self.cost_threshold and not np.isinf(cost):
                        matches.append((row_idx, col_idx))
                        self.logger.debug(f"匹配轨迹{active_regions[row_idx].id}与区域{col_idx}, 成本: {cost:.2f}")
                
                # 如果没有找到任何匹配，使用备用算法
                if not matches and self.fallback_enabled:
                    self.logger.warning("匈牙利算法未找到有效匹配，使用备用算法")
                    return self._fallback_greedy_matching(current_regions, active_regions, distance_threshold)
                
                # 如果启用重连，尝试重连未匹配的区域
                if self.enable_reconnection and len(matches) < len(active_regions):
                    reconnection_matches = self._attempt_hungarian_reconnection(
                        current_regions, active_regions, matches, frame_idx
                    )
                    matches.extend(reconnection_matches)
                
                self.logger.debug(f"匈牙利算法第{frame_idx}帧: 匹配{len(matches)}对区域")
                return matches
                
            except Exception as e:
                self.logger.warning(f"匈牙利算法求解失败: {e}，使用备用贪婪算法")
                return self._fallback_greedy_matching(current_regions, active_regions, distance_threshold)
            
        except Exception as e:
            self.logger.error(f"匈牙利算法匹配失败: {e}")
            return self._fallback_greedy_matching(current_regions, active_regions, distance_threshold)
    
    def _build_robust_cost_matrix(self, current_regions: List[Dict], 
                                  active_regions: List, distance_threshold: float) -> np.ndarray:
        """构建鲁棒的成本矩阵"""
        try:
            n_tracked = len(active_regions)
            n_current = len(current_regions)
            
            # 初始化成本矩阵
            cost_matrix = np.full((n_tracked, n_current), np.inf)
            
            # 预计算距离矩阵
            tracked_centers = np.array([r.trajectory[-1] for r in active_regions])
            current_centers = np.array([r['center'] for r in current_regions])
            
            distances = cdist(tracked_centers, current_centers)
            
            # 计算各种成本组件
            for i, tracked_region in enumerate(active_regions):
                for j, current_region in enumerate(current_regions):
                    distance = distances[i, j]
                    
                    # 如果距离超过阈值，跳过（保持为inf）
                    if distance > distance_threshold:
                        continue
                    
                    # 计算综合成本
                    cost = self._calculate_robust_assignment_cost(
                        tracked_region, current_region, distance, distance_threshold
                    )
                    cost_matrix[i, j] = cost
            
            return cost_matrix
            
        except Exception as e:
            self.logger.error(f"鲁棒成本矩阵构建失败: {e}")
            return np.array([])
    
    def _calculate_robust_assignment_cost(self, tracked_region, current_region, 
                                        distance: float, distance_threshold: float) -> float:
        """计算鲁棒的分配成本"""
        try:
            # 1. 基础距离成本（标准化）
            distance_cost = distance / distance_threshold
            
            if not self.use_weighted_cost:
                return distance_cost
            
            # 2. 强度差异成本（标准化并限制范围）
            tracked_intensity = getattr(tracked_region, 'intensity', 0)
            current_intensity = current_region.get('intensity', 0)
            
            if tracked_intensity > 0 or current_intensity > 0:
                max_intensity = max(tracked_intensity, current_intensity, 0.1)
                intensity_diff = abs(tracked_intensity - current_intensity)
                intensity_cost = (intensity_diff / max_intensity) * 0.3  # 降低权重
            else:
                intensity_cost = 0.1  # 默认小成本
            
            # 3. 面积差异成本（标准化并限制范围）
            tracked_area = getattr(tracked_region, 'area', 0)
            current_area = current_region.get('area', 0)
            
            if tracked_area > 0 or current_area > 0:
                max_area = max(tracked_area, current_area, 1.0)
                area_diff = abs(tracked_area - current_area)
                area_cost = (area_diff / max_area) * 0.2  # 降低权重
            else:
                area_cost = 0.1  # 默认小成本
            
            # 4. 速度一致性成本（可选，基于轨迹历史）
            velocity_cost = 0
            if len(tracked_region.trajectory) >= 2:
                try:
                    prev_pos = np.array(tracked_region.trajectory[-2])
                    curr_pos = np.array(tracked_region.trajectory[-1])
                    new_pos = np.array(current_region['center'])
                    
                    prev_velocity = curr_pos - prev_pos
                    new_velocity = new_pos - curr_pos
                    
                    velocity_diff = np.linalg.norm(new_velocity - prev_velocity)
                    velocity_cost = min(0.3, velocity_diff / distance_threshold * 0.1)  # 限制最大值
                except:
                    velocity_cost = 0.05  # 默认小成本
            
            # 5. 非活跃帧数惩罚（标准化）
            inactive_penalty = min(0.2, tracked_region.inactive_frames / self.max_inactive_frames * 0.1)
            
            # 综合成本（确保所有组件都是有限值）
            total_cost = (distance_cost + intensity_cost + area_cost + 
                         velocity_cost + inactive_penalty)
            
            # 确保成本是有限的正值
            total_cost = max(0.01, min(total_cost, distance_threshold * 2))
            
            return total_cost
            
        except Exception as e:
            self.logger.warning(f"成本计算失败: {e}")
            # 返回基于距离的简单成本
            return max(0.01, min(distance, distance_threshold * 2))
    
    def _is_matrix_feasible(self, cost_matrix: np.ndarray) -> bool:
        """检查成本矩阵是否可行"""
        try:
            if cost_matrix.size == 0:
                return False
            
            # 检查是否所有行或列都是无穷大
            finite_values = np.isfinite(cost_matrix)
            
            # 每行至少要有一个有限值
            rows_with_finite = np.any(finite_values, axis=1)
            # 每列至少要有一个有限值
            cols_with_finite = np.any(finite_values, axis=0)
            
            # 检查是否有足够的有限值来进行分配
            min_dim = min(cost_matrix.shape)
            finite_count = np.sum(finite_values)
            
            # 至少需要min_dim个有限值，并且每行每列都要有可选项
            feasible = (finite_count >= min_dim and 
                       np.all(rows_with_finite) and 
                       np.all(cols_with_finite))
            
            if not feasible:
                self.logger.debug(f"矩阵不可行: 形状{cost_matrix.shape}, "
                                f"有限值{finite_count}, "
                                f"可行行数{np.sum(rows_with_finite)}, "
                                f"可行列数{np.sum(cols_with_finite)}")
            
            return feasible
            
        except Exception as e:
            self.logger.warning(f"可行性检查失败: {e}")
            return False
    
    def _adjust_cost_matrix(self, cost_matrix: np.ndarray, distance_threshold: float) -> np.ndarray:
        """自适应调整成本矩阵"""
        try:
            adjusted_matrix = cost_matrix.copy()
            
            # 策略1: 将所有无穷大值替换为大但有限的值
            inf_mask = np.isinf(adjusted_matrix)
            if np.any(inf_mask):
                finite_values = adjusted_matrix[~inf_mask]
                if len(finite_values) > 0:
                    max_finite = np.max(finite_values)
                    replacement_value = max_finite * 2 + distance_threshold
                else:
                    replacement_value = distance_threshold * 3
                
                adjusted_matrix[inf_mask] = replacement_value
                self.logger.debug(f"替换了{np.sum(inf_mask)}个无穷大值")
            
            # 策略2: 确保每行每列都有可接受的选项
            # 如果某行所有值都太大，找到该行的最小值并适当减小
            for i in range(adjusted_matrix.shape[0]):
                row_min = np.min(adjusted_matrix[i, :])
                if row_min > self.cost_threshold:
                    # 将该行最小值调整为可接受范围
                    min_col = np.argmin(adjusted_matrix[i, :])
                    adjusted_matrix[i, min_col] = self.cost_threshold * 0.9
                    self.logger.debug(f"调整行{i}的最小值")
            
            # 策略3: 类似地处理列
            for j in range(adjusted_matrix.shape[1]):
                col_min = np.min(adjusted_matrix[:, j])
                if col_min > self.cost_threshold:
                    min_row = np.argmin(adjusted_matrix[:, j])
                    adjusted_matrix[min_row, j] = self.cost_threshold * 0.9
                    self.logger.debug(f"调整列{j}的最小值")
            
            return adjusted_matrix
            
        except Exception as e:
            self.logger.warning(f"成本矩阵调整失败: {e}")
            return cost_matrix
    
    def _attempt_hungarian_reconnection(self, current_regions: List[Dict], 
                                      active_regions: List,
                                      existing_matches: List[Tuple[int, int]], 
                                      frame_idx: int) -> List[Tuple[int, int]]:
        """使用匈牙利算法尝试重连（增强版）"""
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
            reconnection_threshold = self.distance_threshold * 2.5
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
                        # 重连时的成本包含非活跃惩罚但更宽松
                        reconnection_cost = distance + tracked_region.inactive_frames * 2
                        cost_matrix[i, j] = reconnection_cost
            
            # 如果有可重连的组合，使用匈牙利算法
            if self._is_matrix_feasible(cost_matrix):
                try:
                    row_indices, col_indices = linear_sum_assignment(cost_matrix)
                    
                    for row_idx, col_idx in zip(row_indices, col_indices):
                        cost = cost_matrix[row_idx, col_idx]
                        if cost < reconnection_threshold * 1.5:  # 更宽松的阈值
                            original_tracked_idx = unmatched_tracked[row_idx]
                            original_current_idx = unmatched_current[col_idx]
                            
                            reconnection_matches.append((original_tracked_idx, original_current_idx))
                            
                            region = active_regions[original_tracked_idx]
                            self.logger.debug(f"重连轨迹{region.id}: 成本{cost:.2f}")
                            
                except Exception as e:
                    self.logger.warning(f"重连匈牙利算法失败: {e}")
            else:
                self.logger.debug("重连成本矩阵不可行，跳过重连")
            
        except Exception as e:
            self.logger.warning(f"重连尝试失败: {e}")
        
        return reconnection_matches
    
    def _fallback_greedy_matching(self, current_regions: List[Dict], 
                                active_regions: List, 
                                distance_threshold: float) -> List[Tuple[int, int]]:
        """增强的备用贪婪匹配算法"""
        try:
            if not current_regions or not active_regions:
                return []
            
            current_centers = np.array([r['center'] for r in current_regions])
            tracked_centers = np.array([r.trajectory[-1] for r in active_regions])
            
            distances = cdist(tracked_centers, current_centers)
            
            matches = []
            used_current = set()
            used_tracked = set()
            
            # 获取所有距离的排序索引
            dist_indices = np.unravel_index(np.argsort(distances.ravel()), distances.shape)
            
            for tracked_idx, current_idx in zip(dist_indices[0], dist_indices[1]):
                # 跳过已使用的区域
                if tracked_idx in used_tracked or current_idx in used_current:
                    continue
                
                # 检查距离是否在阈值内
                if distances[tracked_idx, current_idx] < distance_threshold:
                    matches.append((tracked_idx, current_idx))
                    used_tracked.add(tracked_idx)
                    used_current.add(current_idx)
                else:
                    # 距离太远，停止搜索
                    break
            
            self.logger.debug(f"备用贪婪算法找到{len(matches)}个匹配")
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
                "鲁棒成本矩阵",
                "自适应阈值调整",
                "智能备用机制"
            ],
            'advantages': [
                "全局最优匹配",
                "自适应处理困难情况",
                "多层备用保障",
                "提高成功率"
            ],
            'disadvantages': [
                "计算复杂度较高",
                "内存消耗较大",
                "参数调优复杂"
            ],
            'improvements': [
                "增强的成本矩阵构建",
                "自适应阈值调整",
                "智能可行性检查",
                "鲁棒的备用机制"
            ],
            'best_for': [
                "高精度要求场景",
                "复杂跟踪问题",
                "质量优先应用",
                "研究分析场景"
            ]
        })
        return info