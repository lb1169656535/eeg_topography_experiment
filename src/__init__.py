# EEG脑电地形图运动轨迹分析包
__version__ = "1.0.0"
__author__ = "EEG Research Team"

from .data_loader import EEGDataLoader
from .topography import TopographyGenerator
from .trajectory_analysis import TrajectoryAnalyzer
from .visualization import Visualizer

__all__ = [
    'EEGDataLoader',
    'TopographyGenerator', 
    'TrajectoryAnalyzer',
    'Visualizer'
]