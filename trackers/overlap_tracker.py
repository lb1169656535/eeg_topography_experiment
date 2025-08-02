import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Tuple, Dict
from .base_tracker import BaseTracker

class OverlapTracker(BaseTracker):
    """重叠度匹配跟踪器"""
    
    def __init__(self, config):
        super().__init__(config)
        self.algorithm_name = "overlap"
        self.description = "重叠度匹配 - 基于区域重叠"
        
        # 获取算法特定配置
        alg_config = config.get_algorithm_config('overlap')
        self.overlap_threshold = alg_config.get('overlap_threshold', 0.3)
        self.distance_threshold = alg_config.get('distance_threshold', 35.0)
        self.enable_reconnection = alg_config.get('enable_reconnection', True)
        self.max_inactive_frames = alg_config.get('max_inactive_frames', 20)
        
        self.algorithm_params = {
            'overlap_threshold': self.overlap_threshold,
            'distance_threshold': self.distance_threshold,
            'enable_reconnection': self.enable_reconnection,
            'max_inactive_frames': self.max_inactive_frames
        }
    
    def match_regions(self, current_regions: List[Dict], 
                     distance_threshold: float = None, frame_idx: int = 0) -> List[Tuple[int, int]]:
        """重叠度匹配算法"""
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
            matches = []
            used_current = set()
            used_tracked = set()
            
            # 为每个跟踪区域寻找最佳匹配
            for i, tracked_region in enumerate(active_regions):
                if i in used_tracked:
                    continue
                
                best_match = -1
                best_score = 0
                
                for j, current_region in enumerate(current_regions):
                    if j in used_current:
                        continue
                    
                    # 计算综合匹配分数
                    score = self._calculate_match_score(tracked_region, current_region, distance_threshold)
                    
                    if score > best_score and score > self.overlap_threshold:
                        best_score = score
                        best_match = j
                
                if best_match >= 0:
                    matches.append((i, best_match))
                    used_tracked.add(i)
                    used_current.add(best_match)
                    
                    self.logger.debug(f"重叠匹配轨迹{tracked_region.id}: 分数{best_score:.3f}")
            
            # 如果启用重连，尝试重连未匹配的区域
            if self.enable_reconnection and len(matches) < len(active_regions):
                reconnection_matches = self._attempt_overlap_reconnection(
                    current_regions, active_regions, used_current, used_tracked, frame_idx
                )
                matches.extend(reconnection_matches)
            
            self.logger.debug(f"重叠算法第{frame_idx}帧: 匹配{len(matches)}对区域")
            return matches
            
        except Exception as e:
            self.logger.error(f"重叠匹配失败: {e}")
            return []
    
    def _calculate_match_score(self, tracked_region, current_region, distance_threshold: float) -> float:
        """计算匹配分数"""
        try:
            tracked_center = np.array(tracked_region.trajectory[-1])
            current_center = np.array(current_region['center'])
            
            # 1. 距离分数
            distance = np.linalg.norm(tracked_center - current_center)
            if distance > distance_threshold:
                return 0.0
            
            distance_score = 1.0 - (distance / distance_threshold)
            
            # 2. 空间重叠分数（简化版本，基于距离和大小）
            overlap_score = self._estimate_spatial_overlap(tracked_region, current_region, distance)
            
            # 3. 强度相似性分数
            intensity_score = self._calculate_intensity_similarity(tracked_region, current_region)
            
            # 4. 面积相似性分数
            area_score = self._calculate_area_similarity(tracked_region, current_region)
            
            # 5. 形状稳定性分数
            stability_score = self._calculate_stability_score(tracked_region)
            
            # 综合分数（权重可调）
            total_score = (distance_score * 0.25 + 
                         overlap_score * 0.35 + 
                         intensity_score * 0.15 + 
                         area_score * 0.15 + 
                         stability_score * 0.1)
            
            return total_score
            
        except Exception as e:
            self.logger.warning(f"匹配分数计算失败: {e}")
            return 0.0
    
    def _estimate_spatial_overlap(self, tracked_region, current_region, distance: float) -> float:
        """估算空间重叠度"""
        try:
            # 简化的重叠估算，基于距离和区域大小
            tracked_area = getattr(tracked_region, 'area', 0)
            current_area = current_region.get('area', 0)
            
            if tracked_area == 0 or current_area == 0:
                return 0.0
            
            # 估算重叠区域：距离越近，重叠度越高
            estimated_radius_tracked = np.sqrt(tracked_area / np.pi)
            estimated_radius_current = np.sqrt(current_area / np.pi)
            
            # 如果两个圆心距离小于两个半径之和，则有重叠
            radius_sum = estimated_radius_tracked + estimated_radius_current
            
            if distance >= radius_sum:
                return 0.0
            elif distance <= abs(estimated_radius_tracked - estimated_radius_current):
                # 一个完全包含另一个
                return 1.0
            else:
                # 部分重叠，使用简化公式
                overlap_ratio = 1.0 - (distance / radius_sum)
                return max(0.0, min(1.0, overlap_ratio))
                
        except Exception as e:
            self.logger.warning(f"重叠估算失败: {e}")
            return 0.0
    
    def _calculate_intensity_similarity(self, tracked_region, current_region) -> float:
        """计算强度相似性"""
        try:
            tracked_intensity = getattr(tracked_region, 'intensity', 0)
            current_intensity = current_region.get('intensity', 0)
            
            if tracked_intensity == 0 and current_intensity == 0:
                return 1.0
            
            max_intensity = max(tracked_intensity, current_intensity, 0.1)
            intensity_diff = abs(tracked_intensity - current_intensity)
            
            similarity = 1.0 - min(1.0, intensity_diff / max_intensity)
            return similarity
            
        except Exception as e:
            self.logger.warning(f"强度相似性计算失败: {e}")
            return 0.0
    
    def _calculate_area_similarity(self, tracked_region, current_region) -> float:
        """计算面积相似性"""
        try:
            tracked_area = getattr(tracked_region, 'area', 0)
            current_area = current_region.get('area', 0)
            
            if tracked_area == 0 and current_area == 0:
                return 1.0
            
            max_area = max(tracked_area, current_area, 1.0)
            area_diff = abs(tracked_area - current_area)
            
            similarity = 1.0 - min(1.0, area_diff / max_area)
            return similarity
            
        except Exception as e:
            self.logger.warning(f"面积相似性计算失败: {e}")
            return 0.0
    
    def _calculate_stability_score(self, tracked_region) -> float:
        """计算稳定性分数"""
        try:
            # 基于轨迹长度和非活跃帧数的稳定性
            trajectory_length = len(tracked_region.trajectory)
            inactive_frames = getattr(tracked_region, 'inactive_frames', 0)
            
            # 轨迹越长越稳定
            length_score = min(1.0, trajectory_length / 20.0)
            
            # 非活跃帧数越少越稳定
            activity_score = max(0.0, 1.0 - inactive_frames / self.max_inactive_frames)
            
            stability = (length_score * 0.6 + activity_score * 0.4)
            return stability
            
        except Exception as e:
            self.logger.warning(f"稳定性分数计算失败: {e}")
            return 0.5
    
    def _attempt_overlap_reconnection(self, current_regions: List[Dict], 
                                    active_regions: List,
                                    used_current: set, used_tracked: set, 
                                    frame_idx: int) -> List[Tuple[int, int]]:
        """尝试重叠重连"""
        reconnection_matches = []
        
        try:
            # 获取未匹配的区域
            unmatched_tracked = [i for i in range(len(active_regions)) if i not in used_tracked]
            unmatched_current = [i for i in range(len(current_regions)) if i not in used_current]
            
            if not unmatched_tracked or not unmatched_current:
                return reconnection_matches
            
            # 使用更宽松的阈值进行重连
            reconnection_threshold = self.overlap_threshold * 0.7
            
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
                    score = self._calculate_match_score(tracked_region, current_region, 
                                                      self.distance_threshold * 1.5)
                    
                    if score > reconnection_threshold and score > best_score:
                        best_score = score
                        best_match = current_idx
                
                if best_match >= 0:
                    reconnection_matches.append((tracked_idx, best_match))
                    unmatched_current.remove(best_match)
                    
                    self.logger.debug(f"重叠重连轨迹{tracked_region.id}: 分数{best_score:.3f}")
            
        except Exception as e:
            self.logger.warning(f"重叠重连失败: {e}")
        
        return reconnection_matches
    
    def get_algorithm_info(self) -> Dict:
        """获取算法信息"""
        info = super().get_algorithm_info()
        info.update({
            'description': self.description,
            'characteristics': [
                "空间重叠分析",
                "多特征综合评分",
                "形状感知匹配",
                "稳定性优化"
            ],
            'advantages': [
                "考虑区域形状信息",
                "对形变有一定容忍度",
                "匹配精度较高",
                "能处理部分遮挡"
            ],
            'disadvantages': [
                "计算复杂度较高",
                "依赖准确的区域检测",
                "对噪声敏感",
                "参数较多"
            ],
            'best_for': [
                "形状稳定的目标",
                "需要精确匹配的场景",
                "有重叠可能的场景",
                "区域边界清晰的场景"
            ]
        })
        return info