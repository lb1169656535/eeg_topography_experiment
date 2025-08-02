"""
跟踪器工厂类
用于创建和管理不同类型的跟踪算法
"""

import logging
from typing import Dict, List, Optional
from .base_tracker import BaseTracker
from .greedy_tracker import GreedyTracker
from .hungarian_tracker import HungarianTracker
from .kalman_tracker import KalmanTracker
from .overlap_tracker import OverlapTracker
from .hybrid_tracker import HybridTracker

class TrackerFactory:
    """跟踪器工厂类"""
    
    # 注册的跟踪器类
    _trackers = {
        'greedy': GreedyTracker,
        'hungarian': HungarianTracker, 
        'kalman': KalmanTracker,
        'overlap': OverlapTracker,
        'hybrid': HybridTracker
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @classmethod
    def create_tracker(cls, algorithm_name: str, config) -> Optional[BaseTracker]:
        """创建指定类型的跟踪器"""
        logger = logging.getLogger(__name__)
        
        if algorithm_name not in cls._trackers:
            logger.error(f"未知的跟踪算法: {algorithm_name}")
            return None
        
        try:
            tracker_class = cls._trackers[algorithm_name]
            tracker = tracker_class(config)
            logger.info(f"成功创建{algorithm_name}跟踪器")
            return tracker
        except Exception as e:
            logger.error(f"创建{algorithm_name}跟踪器失败: {e}")
            import traceback
            logger.debug(f"详细错误信息: {traceback.format_exc()}")
            return None
    
    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        """获取可用的算法列表"""
        return list(cls._trackers.keys())
    
    @classmethod
    def get_algorithm_info(cls, algorithm_name: str) -> Dict:
        """获取算法信息"""
        if algorithm_name not in cls._trackers:
            return {'error': f'未知算法: {algorithm_name}'}
        
        try:
            # 创建临时实例获取信息
            from config import Config
            temp_tracker = cls._trackers[algorithm_name](Config)
            return temp_tracker.get_algorithm_info()
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"获取{algorithm_name}算法信息失败: {e}")
            return {
                'name': algorithm_name,
                'description': cls._trackers[algorithm_name].__doc__ or f'{algorithm_name} 跟踪算法',
                'error': f'获取算法信息失败: {e}'
            }
    
    @classmethod
    def compare_algorithms(cls, config) -> Dict:
        """比较所有可用算法"""
        comparison = {
            'available_algorithms': cls.get_available_algorithms(),
            'algorithm_details': {}
        }
        
        for algorithm in cls.get_available_algorithms():
            info = cls.get_algorithm_info(algorithm)
            comparison['algorithm_details'][algorithm] = info
        
        return comparison
    
    @classmethod
    def register_tracker(cls, algorithm_name: str, tracker_class):
        """注册新的跟踪器类"""
        if not issubclass(tracker_class, BaseTracker):
            raise ValueError("跟踪器类必须继承自BaseTracker")
        
        cls._trackers[algorithm_name] = tracker_class
        logging.getLogger(__name__).info(f"注册新跟踪器: {algorithm_name}")
    
    @classmethod
    def create_all_trackers(cls, config, algorithms: Optional[List[str]] = None) -> Dict[str, BaseTracker]:
        """创建多个跟踪器"""
        if algorithms is None:
            algorithms = config.COMPARISON_ALGORITHMS
        
        trackers = {}
        
        for algorithm in algorithms:
            tracker = cls.create_tracker(algorithm, config)
            if tracker is not None:
                trackers[algorithm] = tracker
            else:
                logging.getLogger(__name__).warning(f"跳过创建失败的跟踪器: {algorithm}")
        
        return trackers
    
    @classmethod
    def validate_algorithm_config(cls, config) -> Dict[str, bool]:
        """验证算法配置"""
        validation_results = {}
        
        for algorithm in config.COMPARISON_ALGORITHMS:
            try:
                if algorithm in cls._trackers:
                    # 检查配置是否存在
                    alg_config = config.get_algorithm_config(algorithm)
                    validation_results[algorithm] = bool(alg_config)
                else:
                    validation_results[algorithm] = False
            except Exception as e:
                logging.getLogger(__name__).error(f"验证{algorithm}配置失败: {e}")
                validation_results[algorithm] = False
        
        return validation_results