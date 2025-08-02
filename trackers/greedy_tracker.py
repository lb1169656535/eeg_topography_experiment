import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Tuple, Dict
from .base_tracker import BaseTracker

class GreedyTracker(BaseTracker):
    """贪婪匹配跟踪算法"""
    
    def __init__(self, config):
        super().__init__(config)
        self.algorithm_name = "greedy"
        self.description = "贪婪匹配算法 - 快速局部最优解"
        
        # 获取算法特定配置
        alg_config = config.get_algorithm_config('greedy')
        self.distance_threshold = alg_config.get('distance_threshold', 25.0)
        self.enable_reconnection = alg_config.get('enable_reconnection', True)
        self.max_inactive_frames = alg_config.get('max_inactive_frames', 25)
        
        self.algorithm_params = {
            'distance_threshold': self.distance_threshold,
            'enable_reconnection': self.enable_reconnection,
            'max_inactive_frames': self.max_inactive_frames
        }
    
    def match_regions(self, current_regions: List[Dict], 
                     distance_threshold: float = None, frame_idx: int = 0) -> List[Tuple[int, int]]:
        """贪婪匹配算法实现"""
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
            # 获取当前区域中心点和已跟踪区域的最新位置
            current_centers = np.array([r['center'] for r in current_regions])
            tracked_centers = np.array([r.trajectory[-1] for r in active_regions])
            
            # 计算距离矩阵
            distances = cdist(tracked_centers, current_centers)
            
            # 贪婪匹配：按距离从小到大进行匹配
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
                    # 由于按距离排序，后续距离只会更大，可以提前结束
                    break
            
            # 如果启用重连，尝试为未匹配的区域进行重连
            if self.enable_reconnection and len(matches) < len(active_regions):
                reconnection_matches = self._attempt_reconnection(
                    current_regions, active_regions, used_current, used_tracked, frame_idx
                )
                matches.extend(reconnection_matches)
            
            self.logger.debug(f"贪婪算法第{frame_idx}帧: 匹配{len(matches)}对区域")
            
            return matches
            
        except Exception as e:
            self.logger.error(f"贪婪匹配算法失败: {e}")
            return []
    
    def _attempt_reconnection(self, current_regions: List[Dict], 
                            active_regions: List, 
                            used_current: set, used_tracked: set, 
                            frame_idx: int) -> List[Tuple[int, int]]:
        """尝试重新连接断开的轨迹"""
        reconnection_matches = []
        
        try:
            # 扩大搜索距离进行重连
            reconnection_threshold = self.distance_threshold * 1.5
            
            # 获取未匹配的区域
            unmatched_tracked = [i for i in range(len(active_regions)) if i not in used_tracked]
            unmatched_current = [i for i in range(len(current_regions)) if i not in used_current]
            
            if not unmatched_tracked or not unmatched_current:
                return reconnection_matches
            
            # 计算未匹配区域间的距离
            unmatched_tracked_centers = np.array([active_regions[i].trajectory[-1] for i in unmatched_tracked])
            unmatched_current_centers = np.array([current_regions[i]['center'] for i in unmatched_current])
            
            distances = cdist(unmatched_tracked_centers, unmatched_current_centers)
            
            # 贪婪重连匹配
            reconnection_used_current = set()
            reconnection_used_tracked = set()
            
            dist_indices = np.unravel_index(np.argsort(distances.ravel()), distances.shape)
            
            for rel_tracked_idx, rel_current_idx in zip(dist_indices[0], dist_indices[1]):
                if rel_tracked_idx in reconnection_used_tracked or rel_current_idx in reconnection_used_current:
                    continue
                
                if distances[rel_tracked_idx, rel_current_idx] < reconnection_threshold:
                    # 转换回原始索引
                    original_tracked_idx = unmatched_tracked[rel_tracked_idx]
                    original_current_idx = unmatched_current[rel_current_idx]
                    
                    # 检查非活跃帧数，如果太多则不重连
                    region = active_regions[original_tracked_idx]
                    if region.inactive_frames < self.max_inactive_frames:
                        reconnection_matches.append((original_tracked_idx, original_current_idx))
                        reconnection_used_tracked.add(rel_tracked_idx)
                        reconnection_used_current.add(rel_current_idx)
                        
                        self.logger.debug(f"重连轨迹{region.id}: 距离{distances[rel_tracked_idx, rel_current_idx]:.2f}")
            
        except Exception as e:
            self.logger.warning(f"重连尝试失败: {e}")
        
        return reconnection_matches
    
    def compute_match_quality(self, tracked_region, current_region, distance: float) -> float:
        """计算匹配质量分数"""
        try:
            # 距离分数（距离越小分数越高）
            distance_score = max(0, 1.0 - distance / self.distance_threshold)
            
            # 强度相似性分数
            intensity_diff = abs(tracked_region.intensity - current_region.get('intensity', 0))
            max_intensity = max(tracked_region.intensity, current_region.get('intensity', 0), 0.1)
            intensity_score = 1.0 - min(1.0, intensity_diff / max_intensity)
            
            # 面积相似性分数
            area_diff = abs(tracked_region.area - current_region.get('area', 0))
            max_area = max(tracked_region.area, current_region.get('area', 0), 1.0)
            area_score = 1.0 - min(1.0, area_diff / max_area)
            
            # 综合分数
            quality_score = (distance_score * 0.6 + 
                           intensity_score * 0.25 + 
                           area_score * 0.15)
            
            return quality_score
            
        except Exception as e:
            self.logger.warning(f"匹配质量计算失败: {e}")
            return distance_score if 'distance_score' in locals() else 0.0
    
    def get_algorithm_info(self) -> Dict:
        """获取算法信息"""
        info = super().get_algorithm_info()
        info.update({
            'description': self.description,
            'characteristics': [
                "快速计算",
                "局部最优解",
                "贪婪策略",
                "支持轨迹重连"
            ],
            'advantages': [
                "计算效率高",
                "实现简单",
                "内存消耗低",
                "适合实时处理"
            ],
            'disadvantages': [
                "可能陷入局部最优",
                "对噪声敏感",
                "匹配质量不如全局算法"
            ],
            'best_for': [
                "实时处理场景",
                "计算资源有限场景",
                "轨迹数量较少场景"
            ]
        })
        return info