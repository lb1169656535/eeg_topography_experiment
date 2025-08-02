# 跟踪算法模块
"""
EEG轨迹跟踪算法集合
包含多种不同的轨迹跟踪算法实现
"""

from .base_tracker import BaseTracker
from .greedy_tracker import GreedyTracker
from .hungarian_tracker import HungarianTracker
from .kalman_tracker import KalmanTracker
from .overlap_tracker import OverlapTracker
from .hybrid_tracker import HybridTracker
from .tracker_factory import TrackerFactory

__all__ = [
    'BaseTracker',
    'GreedyTracker', 
    'HungarianTracker',
    'KalmanTracker',
    'OverlapTracker',
    'HybridTracker',
    'TrackerFactory'
]

__version__ = "2.0.0"