import os
import numpy as np

class Config:
    # Data path configuration
    DATA_ROOT = "../data/ds005262"
    RESULTS_ROOT = "./results"
    LOGS_ROOT = "./logs"
    
    # Ensure directories exist
    for path in [RESULTS_ROOT, LOGS_ROOT, 
                 os.path.join(RESULTS_ROOT, "topographies"),
                 os.path.join(RESULTS_ROOT, "trajectories"),
                 os.path.join(RESULTS_ROOT, "analysis"),
                 os.path.join(RESULTS_ROOT, "videos"),
                 os.path.join(RESULTS_ROOT, "algorithm_comparison")]:
        os.makedirs(path, exist_ok=True)
    
    # EEG data processing parameters
    SAMPLING_RATE = 500  
    LOW_FREQ = 1.0       
    HIGH_FREQ = 50.0     
    
    # Topography generation parameters
    TOPO_SIZE = (128, 128)
    INTERPOLATION_METHOD = 'cubic'
     
    # 帧数控制参数 - 新增
    MAX_FRAMES_PER_EPOCH = 300          # 每个epoch最多处理的帧数
    MAX_ANIMATION_FRAMES = 300          # 动画最大帧数
    MAX_SAVE_FRAMES = 50                # 保存帧序列的最大帧数
    
    # 实验规模配置 - 支持所有被试
    MAX_SUBJECTS = 12               # 处理所有12个被试
    MAX_EPOCHS_PER_SUBJECT = 3      # 每个被试处理3个epoch
    MAX_SESSIONS_PER_SUBJECT = 5    # 每个被试最多处理5个session
    MEMORY_LIMIT_MB = 4096          # 增加内存限制

    
    # Algorithm comparison configuration
    ENABLE_ALGORITHM_COMPARISON = True
    COMPARISON_ALGORITHMS = [
        'greedy',           # Greedy matching (original default)
        'hungarian',        # Hungarian algorithm
        'kalman',          # Kalman prediction
        'overlap',         # Overlap matching
        'hybrid'           # Hybrid algorithm
    ]
    
    # Target tracking parameters
    THRESHOLD_PERCENTILE = 88
    MIN_REGION_SIZE = 25       
    MAX_REGIONS = 6            
    
    # Trajectory analysis parameters
    TIME_WINDOW = 2.0          
    TRAJECTORY_SMOOTH_FACTOR = 3  
    
    # Visualization parameters
    COLORMAP = 'RdYlBu_r'      
    FPS = 10                   
    DPI = 150                  
    
    # Algorithm-specific parameters
    ALGORITHM_CONFIGS = {
        'greedy': {
            'distance_threshold': 25.0,
            'enable_prediction': False,
            'enable_reconnection': True,
            'max_inactive_frames': 25,
            'description': 'Greedy matching algorithm - fast local optimum'
        },
        'hungarian': {
            'distance_threshold': 35.0,        # 增加阈值
            'enable_reconnection': True,
            'max_inactive_frames': 30,         # 增加容忍度
            'adaptive_threshold': True,        # 新增
            'fallback_enabled': True,          # 新增
            'description': 'Hungarian algorithm - enhanced version'
        },   
        'kalman': {
            'distance_threshold': 30.0,
            'enable_prediction': True,
            'prediction_weight': 0.4,
            'enable_reconnection': True,
            'max_inactive_frames': 30,
            'description': 'Kalman prediction algorithm - motion-based prediction'
        },
        'overlap': {
            'overlap_threshold': 0.3,
            'distance_threshold': 35.0,
            'enable_reconnection': True,
            'max_inactive_frames': 20,
            'description': 'Overlap matching - region overlap based'
        },
        'hybrid': {
            'distance_threshold': 25.0,
            'overlap_weight': 0.4,
            'intensity_weight': 0.1,
            'area_weight': 0.1,
            'enable_prediction': True,
            'enable_reconnection': True,
            'max_inactive_frames': 30,
            'description': 'Hybrid algorithm - comprehensive multi-feature'
        }
    }
    
    # Performance evaluation metrics
    EVALUATION_METRICS = [
        'trajectory_count',           
        'average_trajectory_length',  
        'max_trajectory_length',     
        'tracking_continuity',       
        'trajectory_smoothness',     
        'computation_time',          
        'memory_usage',             
        'detection_stability',      
        'trajectory_quality'        
    ]
    
    # Visualization configuration
    VISUALIZATION_CONFIG = {
        'generate_comparison_plots': True,    
        'generate_heatmaps': True,           
        'generate_trajectory_overlays': True, 
        'generate_detailed_reports': True,    
        'save_individual_results': True,     
        'create_summary_animations': False   # Disabled for performance
    }
    
    # Tracking optimization parameters
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
    
    # Visualization optimization parameters
    VISUALIZATION_OPTIMIZATION = {
        'auto_font_detection': False,         # Disabled to avoid Chinese font issues
        'fallback_to_english': True,         
        'max_animation_frames': MAX_ANIMATION_FRAMES,
        'save_frame_sequence_fallback': True, 
        'trajectory_alpha': 0.8,             
        'show_direction_arrows': True,       
        'legend_max_items': 10               
    }
    
    # Standard electrode positions
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
        
        # Additional electrodes
        'AF7': (-0.5, 0.65), 'AF3': (-0.25, 0.65), 'AFz': (0, 0.65), 'AF4': (0.25, 0.65), 'AF8': (0.5, 0.65),
        'F5': (-0.55, 0.4), 'F1': (-0.2, 0.4), 'F2': (0.2, 0.4), 'F6': (0.55, 0.4),
        'FT9': (-0.9, 0.2), 'FT7': (-0.75, 0.2), 'FT8': (0.75, 0.2), 'FT10': (0.9, 0.2),
        'C5': (-0.55, 0), 'C1': (-0.2, 0), 'C2': (0.2, 0), 'C6': (0.55, 0),
        'TP9': (-0.9, -0.2), 'TP7': (-0.75, -0.2), 'TP8': (0.75, -0.2), 'TP10': (0.9, -0.2),
        'P5': (-0.55, -0.4), 'P1': (-0.2, -0.4), 'P2': (0.2, -0.4), 'P6': (0.55, -0.4),
        
        # Case variants
        'FP1': (-0.3, 0.85), 'FP2': (0.3, 0.85),
        'fp1': (-0.3, 0.85), 'fp2': (0.3, 0.85), 'fpz': (0, 0.9),
        'f7': (-0.7, 0.4), 'f3': (-0.4, 0.4), 'fz': (0, 0.4), 'f4': (0.4, 0.4), 'f8': (0.7, 0.4),
        't7': (-0.85, 0), 'c3': (-0.4, 0), 'cz': (0, 0), 'c4': (0.4, 0), 't8': (0.85, 0),
        'p7': (-0.7, -0.4), 'p3': (-0.4, -0.4), 'pz': (0, -0.4), 'p4': (0.4, -0.4), 'p8': (0.7, -0.4),
        'o1': (-0.3, -0.85), 'oz': (0, -0.9), 'o2': (0.3, -0.85),
    }
    
    # Frame control methods
    @classmethod
    def get_max_frames(cls, frame_type: str = 'epoch') -> int:
        """Get maximum frame limit"""
        frame_limits = {
            'epoch': cls.MAX_FRAMES_PER_EPOCH,
            'animation': cls.MAX_ANIMATION_FRAMES,
            'save': cls.MAX_SAVE_FRAMES
        }
        return frame_limits.get(frame_type, cls.MAX_FRAMES_PER_EPOCH)
    
    @classmethod
    def set_max_frames(cls, max_frames: int, frame_type: str = 'epoch'):
        """Dynamically set maximum frames"""
        if frame_type == 'epoch':
            cls.MAX_FRAMES_PER_EPOCH = max_frames
        elif frame_type == 'animation':
            cls.MAX_ANIMATION_FRAMES = max_frames
        elif frame_type == 'save':
            cls.MAX_SAVE_FRAMES = max_frames
        
        # Sync related configuration
        cls.VISUALIZATION_OPTIMIZATION['max_animation_frames'] = cls.MAX_ANIMATION_FRAMES
    
    @staticmethod
    def get_default_electrode_position(ch_name: str, n_channels: int, ch_index: int):
        """Generate default position for unknown electrodes"""
        angle = 2 * np.pi * ch_index / n_channels
        radius = 0.7
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        return (x, y)
    
    @classmethod
    def get_algorithm_config(cls, algorithm_name: str):
        """Get specific algorithm configuration"""
        return cls.ALGORITHM_CONFIGS.get(algorithm_name, cls.ALGORITHM_CONFIGS['greedy'])
    
    @classmethod
    def get_experiment_summary(cls):
        """Get experiment configuration summary"""
        return {
            'total_subjects': cls.MAX_SUBJECTS,
            'algorithms_count': len(cls.COMPARISON_ALGORITHMS),
            'algorithm_names': cls.COMPARISON_ALGORITHMS,
            'metrics_count': len(cls.EVALUATION_METRICS),
            'max_epochs_per_subject': cls.MAX_EPOCHS_PER_SUBJECT,
            'max_sessions_per_subject': cls.MAX_SESSIONS_PER_SUBJECT,
            'max_frames_per_epoch': cls.MAX_FRAMES_PER_EPOCH,
            'algorithm_comparison_enabled': cls.ENABLE_ALGORITHM_COMPARISON
        }
    
    @classmethod
    def auto_adjust_parameters(cls, data_characteristics: dict):
        """Auto-adjust parameters based on data characteristics"""
        if 'signal_strength' in data_characteristics:
            signal_strength = data_characteristics['signal_strength']
            
            if signal_strength < 0.3:  # Weak signal
                cls.THRESHOLD_PERCENTILE = 85
                cls.MIN_REGION_SIZE = 15
                print("✓ Detected weak signal, adjusted to high sensitivity parameters")
                
            elif signal_strength > 0.8:  # Strong signal
                cls.THRESHOLD_PERCENTILE = 92
                cls.MIN_REGION_SIZE = 40
                print("✓ Detected strong signal, adjusted to high precision parameters")
        
        if 'noise_level' in data_characteristics:
            noise_level = data_characteristics['noise_level']
            
            if noise_level > 0.6:  # High noise
                cls.TRAJECTORY_SMOOTH_FACTOR = 5
                print("✓ Detected high noise, enabled strong smoothing")
            elif noise_level < 0.2:  # Low noise
                cls.TRAJECTORY_SMOOTH_FACTOR = 2
                print("✓ Detected low noise, enabled fine tracking")
    
    @classmethod
    def get_config_summary(cls):
        """Get current configuration summary"""
        summary = {
            'detection_sensitivity': 'High' if cls.THRESHOLD_PERCENTILE < 90 else 'Medium' if cls.THRESHOLD_PERCENTILE < 95 else 'Low',
            'tracking_aggressiveness': 'High' if cls.TRACKING_OPTIMIZATION['base_distance_threshold'] > 25 else 'Medium',
            'quality_filter': 'Strict' if cls.TRACKING_OPTIMIZATION['quality_threshold'] > 0.25 else 'Lenient',
            'smoothing_level': 'High' if cls.TRAJECTORY_SMOOTH_FACTOR > 4 else 'Medium' if cls.TRAJECTORY_SMOOTH_FACTOR > 2 else 'Low',
            'algorithm_comparison': 'Enabled' if cls.ENABLE_ALGORITHM_COMPARISON else 'Disabled',
            'algorithms_to_compare': len(cls.COMPARISON_ALGORITHMS),
            'max_frames_per_epoch': cls.MAX_FRAMES_PER_EPOCH
        }
        return summary