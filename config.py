import os
import numpy as np

class Config:
    # 数据路径配置
    DATA_ROOT = "../data/ds005262"
    RESULTS_ROOT = "./results"
    LOGS_ROOT = "./logs"
    
    # 确保目录存在
    for path in [RESULTS_ROOT, LOGS_ROOT, 
                 os.path.join(RESULTS_ROOT, "topographies"),
                 os.path.join(RESULTS_ROOT, "trajectories"),
                 os.path.join(RESULTS_ROOT, "analysis"),
                 os.path.join(RESULTS_ROOT, "videos"),
                 os.path.join(RESULTS_ROOT, "algorithm_comparison")]:  # 新增
        os.makedirs(path, exist_ok=True)
    
    # EEG数据处理参数
    SAMPLING_RATE = 500  
    LOW_FREQ = 1.0       
    HIGH_FREQ = 50.0     
    
    # 地形图生成参数
    TOPO_SIZE = (128, 128)
    INTERPOLATION_METHOD = 'cubic'
    
    # 帧数控制参数 - 新增
    MAX_FRAMES_PER_EPOCH = 300        # 每个epoch最多处理的帧数
    MAX_ANIMATION_FRAMES = 300        # 动画最大帧数
    MAX_SAVE_FRAMES = 50              # 保存帧序列的最大帧数
    
    # 实验规模配置 - 支持所有被试
    MAX_SUBJECTS = 12              # 处理所有12个被试
    MAX_EPOCHS_PER_SUBJECT = 3     # 每个被试处理3个epoch
    MAX_SESSIONS_PER_SUBJECT = 5   # 每个被试最多处理5个session
    MEMORY_LIMIT_MB = 4096         # 增加内存限制
    
    # 算法对比配置
    ENABLE_ALGORITHM_COMPARISON = True    # 启用算法对比
    COMPARISON_ALGORITHMS = [             # 要对比的算法
        'greedy',           # 贪婪匹配（原默认算法）
        'hungarian',        # 匈牙利算法
        'kalman',          # 卡尔曼预测
        'overlap',         # 重叠度匹配
        'hybrid'           # 混合算法
    ]
    
    # 目标跟踪参数 - 保持原有设置
    THRESHOLD_PERCENTILE = 88
    MIN_REGION_SIZE = 25       
    MAX_REGIONS = 6            
    
    # 轨迹分析参数
    TIME_WINDOW = 2.0          
    TRAJECTORY_SMOOTH_FACTOR = 3  
    
    # 可视化参数
    COLORMAP = 'RdYlBu_r'      
    FPS = 10                   
    DPI = 150                  
    
    # 各算法的具体参数配置
    ALGORITHM_CONFIGS = {
        'greedy': {
            'distance_threshold': 25.0,
            'enable_prediction': False,
            'enable_reconnection': True,
            'max_inactive_frames': 25,
            'description': '贪婪匹配算法 - 快速局部最优'
        },
        'hungarian': {
            'distance_threshold': 25.0,
            'enable_prediction': False,
            'enable_reconnection': True,
            'max_inactive_frames': 25,
            'description': '匈牙利算法 - 全局最优匹配'
        },
        'kalman': {
            'distance_threshold': 30.0,
            'enable_prediction': True,
            'prediction_weight': 0.4,
            'enable_reconnection': True,
            'max_inactive_frames': 30,
            'description': '卡尔曼预测算法 - 基于运动预测'
        },
        'overlap': {
            'overlap_threshold': 0.3,
            'distance_threshold': 35.0,
            'enable_reconnection': True,
            'max_inactive_frames': 20,
            'description': '重叠度匹配 - 基于区域重叠'
        },
        'hybrid': {
            'distance_threshold': 25.0,
            'overlap_weight': 0.4,
            'intensity_weight': 0.1,
            'area_weight': 0.1,
            'enable_prediction': True,
            'enable_reconnection': True,
            'max_inactive_frames': 30,
            'description': '混合算法 - 综合多种特征'
        }
    }
    
    # 性能评估指标
    EVALUATION_METRICS = [
        'trajectory_count',           # 轨迹数量
        'average_trajectory_length',  # 平均轨迹长度
        'max_trajectory_length',     # 最大轨迹长度
        'tracking_continuity',       # 跟踪连续性
        'trajectory_smoothness',     # 轨迹平滑度
        'computation_time',          # 计算时间
        'memory_usage',             # 内存使用
        'detection_stability',      # 检测稳定性
        'trajectory_quality'        # 轨迹质量
    ]
    
    # 可视化配置
    VISUALIZATION_CONFIG = {
        'generate_comparison_plots': True,     # 生成算法对比图
        'generate_heatmaps': True,            # 生成性能热图
        'generate_trajectory_overlays': True,  # 生成轨迹叠加图
        'generate_detailed_reports': True,     # 生成详细报告
        'save_individual_results': True,      # 保存各算法单独结果
        'create_summary_animations': False    # 创建对比动画（耗时）
    }
    
    # 保持原有的跟踪优化参数
    TRACKING_OPTIMIZATION = {
        'enable_adaptive_threshold': True,     
        'enable_prediction': True,             
        'enable_reconnection': True,           
        'base_distance_threshold': 25.0,      
        'max_distance_threshold': 60.0,       
        'reconnection_distance': 40.0,        
        'max_inactive_frames': 25,            
        'prediction_weight': 0.3,             
        'quality_threshold': 0.2,             
        'min_trajectory_length': 3            
    }
    
    # 保持原有的可视化优化参数
    VISUALIZATION_OPTIMIZATION = {
        'auto_font_detection': True,          
        'fallback_to_english': True,         
        'max_animation_frames': MAX_ANIMATION_FRAMES,  # 使用新参数
        'save_frame_sequence_fallback': True, 
        'trajectory_alpha': 0.8,             
        'show_direction_arrows': True,       
        'legend_max_items': 10               
    }
    
    # 标准电极位置 (保持原有)
    ELECTRODE_POSITIONS = {
        'Fp1': (-0.3, 0.85), 'Fp2': (0.3, 0.85), 'Fpz': (0, 0.9),
        'F7': (-0.7, 0.4), 'F3': (-0.4, 0.4), 'Fz': (0, 0.4), 'F4': (0.4, 0.4), 'F8': (0.7, 0.4),
        'FC5': (-0.5, 0.2), 'FC1': (-0.2, 0.2), 'FCz': (0, 0.2), 'FC2': (0.2, 0.2), 'FC6': (0.5, 0.2),
        'T7': (-0.85, 0), 'C3': (-0.4, 0), 'Cz': (0, 0), 'C4': (0.4, 0), 'T8': (0.85, 0),
        'CP5': (-0.5, -0.2), 'CP1': (-0.2, -0.2), 'CPz': (0, -0.2), 'CP2': (0.2, -0.2), 'CP6': (0.5, -0.2),
        'P7': (-0.7, -0.4), 'P3': (-0.4, -0.4), 'Pz': (0, -0.4), 'P4': (0.4, -0.4), 'P8': (0.7, -0.4),
        'PO9': (-0.8, -0.65), 'PO7': (-0.6, -0.65), 'PO3': (-0.25, -0.65), 'POz': (0, -0.65), 
        'PO4': (0.25, -0.65), 'PO8': (0.6, -0.65), 'PO10': (0.8, -0.65),
        'O1': (-0.3, -0.85), 'Oz': (0, -0.9), 'O2': (0.3, -0.85),
        
        # 额外电极
        'AF7': (-0.5, 0.65), 'AF3': (-0.25, 0.65), 'AFz': (0, 0.65), 'AF4': (0.25, 0.65), 'AF8': (0.5, 0.65),
        'F5': (-0.55, 0.4), 'F1': (-0.2, 0.4), 'F2': (0.2, 0.4), 'F6': (0.55, 0.4),
        'FT9': (-0.9, 0.2), 'FT7': (-0.75, 0.2), 'FT8': (0.75, 0.2), 'FT10': (0.9, 0.2),
        'C5': (-0.55, 0), 'C1': (-0.2, 0), 'C2': (0.2, 0), 'C6': (0.55, 0),
        'TP9': (-0.9, -0.2), 'TP7': (-0.75, -0.2), 'TP8': (0.75, -0.2), 'TP10': (0.9, -0.2),
        'P5': (-0.55, -0.4), 'P1': (-0.2, -0.4), 'P2': (0.2, -0.4), 'P6': (0.55, -0.4),
        
        # 大小写变体
        'FP1': (-0.3, 0.85), 'FP2': (0.3, 0.85),
        'fp1': (-0.3, 0.85), 'fp2': (0.3, 0.85), 'fpz': (0, 0.9),
        'f7': (-0.7, 0.4), 'f3': (-0.4, 0.4), 'fz': (0, 0.4), 'f4': (0.4, 0.4), 'f8': (0.7, 0.4),
        't7': (-0.85, 0), 'c3': (-0.4, 0), 'cz': (0, 0), 'c4': (0.4, 0), 't8': (0.85, 0),
        'p7': (-0.7, -0.4), 'p3': (-0.4, -0.4), 'pz': (0, -0.4), 'p4': (0.4, -0.4), 'p8': (0.7, -0.4),
        'o1': (-0.3, -0.85), 'oz': (0, -0.9), 'o2': (0.3, -0.85),
    }
    
    # 帧数控制方法 - 新增
    @classmethod
    def get_max_frames(cls, frame_type: str = 'epoch') -> int:
        """获取最大帧数限制"""
        frame_limits = {
            'epoch': cls.MAX_FRAMES_PER_EPOCH,
            'animation': cls.MAX_ANIMATION_FRAMES,
            'save': cls.MAX_SAVE_FRAMES
        }
        return frame_limits.get(frame_type, cls.MAX_FRAMES_PER_EPOCH)
    
    @classmethod
    def set_max_frames(cls, max_frames: int, frame_type: str = 'epoch'):
        """动态设置最大帧数"""
        if frame_type == 'epoch':
            cls.MAX_FRAMES_PER_EPOCH = max_frames
        elif frame_type == 'animation':
            cls.MAX_ANIMATION_FRAMES = max_frames
        elif frame_type == 'save':
            cls.MAX_SAVE_FRAMES = max_frames
        
        # 同步更新相关配置
        cls.VISUALIZATION_OPTIMIZATION['max_animation_frames'] = cls.MAX_ANIMATION_FRAMES
    
    # 保持原有方法
    @staticmethod
    def get_default_electrode_position(ch_name: str, n_channels: int, ch_index: int):
        """为未知电极生成默认位置"""
        angle = 2 * np.pi * ch_index / n_channels
        radius = 0.7
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        return (x, y)
    
    # 新增方法
    @classmethod
    def get_algorithm_config(cls, algorithm_name: str):
        """获取特定算法的配置"""
        return cls.ALGORITHM_CONFIGS.get(algorithm_name, cls.ALGORITHM_CONFIGS['greedy'])
    
    @classmethod
    def get_experiment_summary(cls):
        """获取实验配置摘要"""
        return {
            'total_subjects': cls.MAX_SUBJECTS,
            'algorithms_count': len(cls.COMPARISON_ALGORITHMS),
            'algorithm_names': cls.COMPARISON_ALGORITHMS,
            'metrics_count': len(cls.EVALUATION_METRICS),
            'max_epochs_per_subject': cls.MAX_EPOCHS_PER_SUBJECT,
            'max_sessions_per_subject': cls.MAX_SESSIONS_PER_SUBJECT,
            'max_frames_per_epoch': cls.MAX_FRAMES_PER_EPOCH,  # 新增
            'algorithm_comparison_enabled': cls.ENABLE_ALGORITHM_COMPARISON
        }
    
    # 保持原有的自动调整参数方法
    @classmethod
    def auto_adjust_parameters(cls, data_characteristics: dict):
        """根据数据特征自动调整参数"""
        if 'signal_strength' in data_characteristics:
            signal_strength = data_characteristics['signal_strength']
            
            if signal_strength < 0.3:  # 弱信号
                cls.THRESHOLD_PERCENTILE = 85
                cls.MIN_REGION_SIZE = 15
                print("✓ 检测到弱信号，已调整为高敏感性参数")
                
            elif signal_strength > 0.8:  # 强信号
                cls.THRESHOLD_PERCENTILE = 92
                cls.MIN_REGION_SIZE = 40
                print("✓ 检测到强信号，已调整为高精度参数")
        
        if 'noise_level' in data_characteristics:
            noise_level = data_characteristics['noise_level']
            
            if noise_level > 0.6:  # 高噪声
                cls.TRAJECTORY_SMOOTH_FACTOR = 5
                print("✓ 检测到高噪声，已启用强平滑")
            elif noise_level < 0.2:  # 低噪声
                cls.TRAJECTORY_SMOOTH_FACTOR = 2
                print("✓ 检测到低噪声，已启用精细追踪")
    
    # 保持原有的配置摘要方法
    @classmethod
    def get_config_summary(cls):
        """获取当前配置摘要"""
        summary = {
            'detection_sensitivity': 'High' if cls.THRESHOLD_PERCENTILE < 90 else 'Medium' if cls.THRESHOLD_PERCENTILE < 95 else 'Low',
            'tracking_aggressiveness': 'High' if cls.TRACKING_OPTIMIZATION['base_distance_threshold'] > 25 else 'Medium',
            'quality_filter': 'Strict' if cls.TRACKING_OPTIMIZATION['quality_threshold'] > 0.25 else 'Lenient',
            'smoothing_level': 'High' if cls.TRAJECTORY_SMOOTH_FACTOR > 4 else 'Medium' if cls.TRAJECTORY_SMOOTH_FACTOR > 2 else 'Low',
            'algorithm_comparison': 'Enabled' if cls.ENABLE_ALGORITHM_COMPARISON else 'Disabled',
            'algorithms_to_compare': len(cls.COMPARISON_ALGORITHMS),
            'max_frames_per_epoch': cls.MAX_FRAMES_PER_EPOCH  # 新增
        }
        return summary