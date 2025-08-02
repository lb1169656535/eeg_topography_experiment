import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict
from .base_tracker import BaseTracker

class HybridTracker(BaseTracker):
    """混合跟踪器 - 综合多种算法优势"""
    
    def __init__(self, config):
        super().__init__(config)
        self.algorithm_name = "hybrid"
        self.description = "混合算法 - 综合多种特征"
        
        # 获取算法特定配置
        alg_config = config.get_algorithm_config('hybrid')
        self.distance_threshold = alg_config.get('distance_threshold', 25.0)
        self.overlap_weight = alg_config.get('overlap_weight', 0.4)
        self.intensity_weight = alg_config.get('intensity_weight', 0.1)
        self.area_weight = alg_config.get('area_weight', 0.1)
        self.enable_prediction = alg_config.get('enable_prediction', True)
        self.enable_reconnection = alg_config.get('enable_reconnection', True)
        self.max_inactive_frames = alg_config.get('max_inactive_frames', 30)
        
        # 预测相关参数
        self.prediction_weight = 0.3
        
        self.algorithm_params = {
            'distance_threshold': self.distance_threshold,
            'overlap_weight': self.overlap_weight,
            'intensity_weight': self.intensity_weight,
            'area_weight': self.area_weight,
            'enable_prediction': self.enable_prediction,
            'enable_reconnection': self.enable_reconnection,
            'max_inactive_frames': self.max_inactive_frames
        }
    
    def match_regions(self, current_regions: List[Dict], 
                     distance_threshold: float = None, frame_idx: int = 0) -> List[Tuple[int, int]]:
        """混合匹配算法"""
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
            # 构建综合得分矩阵
            score_matrix = self._build_hybrid_score_matrix(current_regions, active_regions)
            
            if score_matrix.size == 0:
                return []
            
            # 转换为成本矩阵用于匈牙利算法
            cost_matrix = 1.0 - score_matrix
            
            # 使用匈牙利算法求解最优分配
            try:
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                matches = []
                
                for row_idx, col_idx in zip(row_indices, col_indices):
                    score = score_matrix[row_idx, col_idx]
                    
                    # 只接受得分高于阈值的匹配
                    if score > 0.3:  # 可调阈值
                        matches.append((row_idx, col_idx))
                        self.logger.debug(f"混合匹配轨迹{active_regions[row_idx].id}: "
                                        f"综合得分{score:.3f}")
                
                # 如果启用重连，尝试重连未匹配的区域
                if self.enable_reconnection and len(matches) < len(active_regions):
                    reconnection_matches = self._attempt_hybrid_reconnection(
                        current_regions, active_regions, matches, frame_idx
                    )
                    matches.extend(reconnection_matches)
                
                self.logger.debug(f"混合算法第{frame_idx}帧: 匹配{len(matches)}对区域")
                return matches
                
            except Exception as e:
                self.logger.warning(f"混合匈牙利分配失败: {e}, 使用备用算法")
                return self._fallback_hybrid_matching(current_regions, active_regions, distance_threshold)
            
        except Exception as e:
            self.logger.error(f"混合匹配失败: {e}")
            return []
    
    def _build_hybrid_score_matrix(self, current_regions: List[Dict], active_regions: List) -> np.ndarray:
        """构建混合得分矩阵"""
        try:
            n_tracked = len(active_regions)
            n_current = len(current_regions)
            
            score_matrix = np.zeros((n_tracked, n_current))
            
            for i, tracked_region in enumerate(active_regions):
                for j, current_region in enumerate(current_regions):
                    score = self._calculate_hybrid_score(tracked_region, current_region)
                    score_matrix[i, j] = score
            
            return score_matrix
            
        except Exception as e:
            self.logger.error(f"混合得分矩阵构建失败: {e}")
            return np.array([])
    
    def _calculate_hybrid_score(self, tracked_region, current_region) -> float:
        """计算混合得分"""
        try:
            # 1. 距离得分
            distance_score = self._calculate_distance_score(tracked_region, current_region)
            
            # 2. 预测得分（如果启用）
            prediction_score = 0.0
            if self.enable_prediction:
                prediction_score = self._calculate_prediction_score(tracked_region, current_region)
            
            # 3. 强度相似性得分
            intensity_score = self._calculate_intensity_score(tracked_region, current_region)
            
            # 4. 面积相似性得分
            area_score = self._calculate_area_score(tracked_region, current_region)
            
            # 5. 重叠度得分
            overlap_score = self._calculate_overlap_score(tracked_region, current_region)
            
            # 6. 轨迹质量得分
            quality_score = self._calculate_quality_score(tracked_region)
            
            # 7. 速度一致性得分
            velocity_score = self._calculate_velocity_score(tracked_region, current_region)
            
            # 综合得分（权重可调）
            total_score = (distance_score * 0.25 + 
                         prediction_score * 0.15 + 
                         intensity_score * self.intensity_weight + 
                         area_score * self.area_weight + 
                         overlap_score * self.overlap_weight + 
                         quality_score * 0.1 + 
                         velocity_score * 0.1)
            
            return min(1.0, total_score)
            
        except Exception as e:
            self.logger.warning(f"混合得分计算失败: {e}")
            return 0.0
    
    def _calculate_distance_score(self, tracked_region, current_region) -> float:
        """计算距离得分"""
        try:
            tracked_center = np.array(tracked_region.trajectory[-1])
            current_center = np.array(current_region['center'])
            
            distance = np.linalg.norm(tracked_center - current_center)
            
            if distance > self.distance_threshold:
                return 0.0
            
            return 1.0 - (distance / self.distance_threshold)
            
        except Exception as e:
            self.logger.warning(f"距离得分计算失败: {e}")
            return 0.0
    
    def _calculate_prediction_score(self, tracked_region, current_region) -> float:
        """计算预测得分"""
        try:
            if len(tracked_region.trajectory) < 2:
                return 0.0
            
            # 简单线性预测
            if len(tracked_region.trajectory) == 2:
                velocity = np.array(tracked_region.trajectory[-1]) - np.array(tracked_region.trajectory[-2])
                predicted = np.array(tracked_region.trajectory[-1]) + velocity * self.prediction_weight
            else:
                # 考虑加速度的预测
                current_pos = np.array(tracked_region.trajectory[-1])
                prev_pos = np.array(tracked_region.trajectory[-2])
                prev_prev_pos = np.array(tracked_region.trajectory[-3])
                
                velocity = current_pos - prev_pos
                acceleration = velocity - (prev_pos - prev_prev_pos)
                
                predicted = current_pos + velocity * self.prediction_weight + acceleration * 0.5 * (self.prediction_weight ** 2)
            
            current_center = np.array(current_region['center'])
            prediction_error = np.linalg.norm(predicted - current_center)
            
            # 预测误差越小，得分越高
            max_error = self.distance_threshold
            if prediction_error > max_error:
                return 0.0
            
            return 1.0 - (prediction_error / max_error)
            
        except Exception as e:
            self.logger.warning(f"预测得分计算失败: {e}")
            return 0.0
    
    def _calculate_intensity_score(self, tracked_region, current_region) -> float:
        """计算强度得分"""
        try:
            tracked_intensity = getattr(tracked_region, 'intensity', 0)
            current_intensity = current_region.get('intensity', 0)
            
            if tracked_intensity == 0 and current_intensity == 0:
                return 1.0
            
            max_intensity = max(tracked_intensity, current_intensity, 0.1)
            intensity_diff = abs(tracked_intensity - current_intensity)
            
            return 1.0 - min(1.0, intensity_diff / max_intensity)
            
        except Exception as e:
            self.logger.warning(f"强度得分计算失败: {e}")
            return 0.0
    
    def _calculate_area_score(self, tracked_region, current_region) -> float:
        """计算面积得分"""
        try:
            tracked_area = getattr(tracked_region, 'area', 0)
            current_area = current_region.get('area', 0)
            
            if tracked_area == 0 and current_area == 0:
                return 1.0
            
            max_area = max(tracked_area, current_area, 1.0)
            area_diff = abs(tracked_area - current_area)
            
            return 1.0 - min(1.0, area_diff / max_area)
            
        except Exception as e:
            self.logger.warning(f"面积得分计算失败: {e}")
            return 0.0
    
    def _calculate_overlap_score(self, tracked_region, current_region) -> float:
        """计算重叠度得分"""
        try:
            tracked_center = np.array(tracked_region.trajectory[-1])
            current_center = np.array(current_region['center'])
            distance = np.linalg.norm(tracked_center - current_center)
            
            tracked_area = getattr(tracked_region, 'area', 0)
            current_area = current_region.get('area', 0)
            
            if tracked_area == 0 or current_area == 0:
                return 0.0
            
            # 估算重叠（简化版本）
            estimated_radius_tracked = np.sqrt(tracked_area / np.pi)
            estimated_radius_current = np.sqrt(current_area / np.pi)
            
            radius_sum = estimated_radius_tracked + estimated_radius_current
            
            if distance >= radius_sum:
                return 0.0
            elif distance <= abs(estimated_radius_tracked - estimated_radius_current):
                return 1.0
            else:
                return 1.0 - (distance / radius_sum)
                
        except Exception as e:
            self.logger.warning(f"重叠得分计算失败: {e}")
            return 0.0
    
    def _calculate_quality_score(self, tracked_region) -> float:
        """计算轨迹质量得分"""
        try:
            # 基于轨迹长度和稳定性
            length_score = min(1.0, len(tracked_region.trajectory) / 20.0)
            stability_score = max(0.0, 1.0 - tracked_region.inactive_frames / self.max_inactive_frames)
            
            return (length_score * 0.6 + stability_score * 0.4)
            
        except Exception as e:
            self.logger.warning(f"质量得分计算失败: {e}")
            return 0.5
    
    def _calculate_velocity_score(self, tracked_region, current_region) -> float:
        """计算速度一致性得分"""
        try:
            if len(tracked_region.trajectory) < 2:
                return 0.5
            
            # 计算历史速度
            prev_velocity = np.array(tracked_region.trajectory[-1]) - np.array(tracked_region.trajectory[-2])
            
            # 计算当前速度
            tracked_center = np.array(tracked_region.trajectory[-1])
            current_center = np.array(current_region['center'])
            current_velocity = current_center - tracked_center
            
            # 速度差异
            velocity_diff = np.linalg.norm(current_velocity - prev_velocity)
            
            # 速度差异越小，得分越高
            max_velocity_diff = 20.0  # 可调参数
            return max(0.0, 1.0 - velocity_diff / max_velocity_diff)
            
        except Exception as e:
            self.logger.warning(f"速度得分计算失败: {e}")
            return 0.5
    
    def _attempt_hybrid_reconnection(self, current_regions: List[Dict], 
                                   active_regions: List,
                                   existing_matches: List[Tuple[int, int]], 
                                   frame_idx: int) -> List[Tuple[int, int]]:
        """混合重连尝试"""
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
            for tracked_idx in unmatched_tracked:
                tracked_region = active_regions[tracked_idx]
                
                # 只为非活跃帧数不太多的区域尝试重连
                if tracked_region.inactive_frames >= self.max_inactive_frames:
                    continue
                
                best_match = -1
                best_score = 0
                
                for current_idx in unmatched_current:
                    current_region = current_regions[current_idx]
                    
                    # 计算重连分数（更宽松的条件）
                    score = self._calculate_hybrid_score(tracked_region, current_region)
                    
                    if score > 0.2 and score > best_score:  # 更低的阈值
                        best_score = score
                        best_match = current_idx
                
                if best_match >= 0:
                    reconnection_matches.append((tracked_idx, best_match))
                    unmatched_current.remove(best_match)
                    
                    self.logger.debug(f"混合重连轨迹{tracked_region.id}: 得分{best_score:.3f}")
            
        except Exception as e:
            self.logger.warning(f"混合重连失败: {e}")
        
        return reconnection_matches
    
    def _fallback_hybrid_matching(self, current_regions: List[Dict], 
                                 active_regions: List, 
                                 distance_threshold: float) -> List[Tuple[int, int]]:
        """备用混合匹配"""
        try:
            matches = []
            used_current = set()
            used_tracked = set()
            
            # 计算所有得分对
            score_pairs = []
            for i, tracked_region in enumerate(active_regions):
                for j, current_region in enumerate(current_regions):
                    score = self._calculate_hybrid_score(tracked_region, current_region)
                    if score > 0.3:
                        score_pairs.append((score, i, j))
            
            # 按得分排序
            score_pairs.sort(reverse=True)
            
            # 贪婪选择
            for score, tracked_idx, current_idx in score_pairs:
                if tracked_idx not in used_tracked and current_idx not in used_current:
                    matches.append((tracked_idx, current_idx))
                    used_tracked.add(tracked_idx)
                    used_current.add(current_idx)
            
            return matches
            
        except Exception as e:
            self.logger.error(f"备用混合算法失败: {e}")
            return []
    
    def get_algorithm_info(self) -> Dict:
        """获取算法信息"""
        info = super().get_algorithm_info()
        info.update({
            'description': self.description,
            'characteristics': [
                "多算法融合",
                "综合特征评估",
                "自适应权重",
                "全局优化"
            ],
            'advantages': [
                "综合多种算法优势",
                "适应性强",
                "匹配精度高",
                "鲁棒性好"
            ],
            'disadvantages': [
                "计算复杂度最高",
                "参数众多",
                "调优困难",
                "资源消耗大"
            ],
            'best_for': [
                "复杂跟踪场景",
                "高精度要求",
                "多变环境",
                "离线分析场景"
            ]
        })
        return info