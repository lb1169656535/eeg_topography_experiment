import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import cv2
from typing import Dict, Tuple, List, Optional
import logging
import warnings

# 抑制插值警告
warnings.filterwarnings('ignore', category=RuntimeWarning)

class TopographyGenerator:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 预计算头部掩码以提高效率
        self._head_mask = self.create_head_mask(config.TOPO_SIZE)
        
    def create_head_mask(self, size: Tuple[int, int]) -> np.ndarray:
        """创建头部轮廓掩码"""
        h, w = size
        center = (w // 2, h // 2)
        radius = min(w, h) // 2 - 5
        
        # 创建圆形掩码
        y, x = np.ogrid[:h, :w]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        
        return mask.astype(bool)
    
    def electrode_to_pixel(self, pos: Tuple[float, float], size: Tuple[int, int]) -> Tuple[int, int]:
        """将电极位置转换为像素坐标"""
        x, y = pos
        h, w = size
        
        # 标准化坐标到像素坐标
        # 电极坐标范围 [-1, 1] 映射到像素坐标
        pixel_x = int((x + 1) * w / 2)
        pixel_y = int((1 - y) * h / 2)  # Y轴翻转
        
        # 确保在边界内
        pixel_x = max(0, min(w - 1, pixel_x))
        pixel_y = max(0, min(h - 1, pixel_y))
        
        return pixel_x, pixel_y
    
    def validate_electrode_data(self, eeg_data: np.ndarray, 
                               electrode_positions: Dict[str, Tuple[float, float]],
                               ch_names: List[str]) -> Tuple[np.ndarray, List[Tuple[float, float]], List[str]]:
        """验证和清理电极数据"""
        valid_positions = []
        valid_values = []
        valid_names = []
        
        for i, ch_name in enumerate(ch_names):
            if ch_name in electrode_positions:
                pos = electrode_positions[ch_name]
                
                # 检查位置是否合理
                if abs(pos[0]) <= 1.2 and abs(pos[1]) <= 1.2:  # 允许略微超出标准范围
                    # 检查数据是否有效
                    if i < len(eeg_data) and not np.isnan(eeg_data[i]) and not np.isinf(eeg_data[i]):
                        valid_positions.append(pos)
                        valid_values.append(eeg_data[i])
                        valid_names.append(ch_name)
                    else:
                        self.logger.warning(f"Invalid data for electrode {ch_name}: {eeg_data[i] if i < len(eeg_data) else 'missing'}")
                else:
                    self.logger.warning(f"Invalid position for electrode {ch_name}: {pos}")
            else:
                self.logger.warning(f"No position found for electrode {ch_name}")
        
        return np.array(valid_values), valid_positions, valid_names
    
    def generate_topography(self, eeg_data: np.ndarray, 
                          electrode_positions: Dict[str, Tuple[float, float]],
                          ch_names: List[str]) -> np.ndarray:
        """生成脑电地形图"""
        size = self.config.TOPO_SIZE
        
        # 验证和清理数据
        values, positions, valid_names = self.validate_electrode_data(
            eeg_data, electrode_positions, ch_names
        )
        
        if len(positions) < 3:
            self.logger.warning(f"Not enough valid electrode positions for interpolation: {len(positions)}")
            return np.zeros(size)
        
        positions = np.array(positions)
        
        # 创建插值网格
        xi = np.linspace(-1.2, 1.2, size[1])  # 稍微扩大范围以获得更好的边界效果
        yi = np.linspace(-1.2, 1.2, size[0])
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # 执行插值
        try:
            # 首先尝试cubic插值
            topography = griddata(positions, values, (xi_grid, yi_grid), 
                                method=self.config.INTERPOLATION_METHOD, 
                                fill_value=0)
            
            # 检查插值结果
            if np.all(np.isnan(topography)):
                raise ValueError("Cubic interpolation failed")
                
        except (ValueError, Exception) as e:
            self.logger.warning(f"Cubic interpolation failed: {e}, trying linear interpolation")
            try:
                topography = griddata(positions, values, (xi_grid, yi_grid), 
                                    method='linear', fill_value=0)
            except Exception as e2:
                self.logger.warning(f"Linear interpolation failed: {e2}, trying nearest neighbor")
                topography = griddata(positions, values, (xi_grid, yi_grid), 
                                    method='nearest', fill_value=0)
        
        # 处理NaN值
        topography = np.nan_to_num(topography, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 应用头部掩码
        topography[~self._head_mask] = 0
        
        # 平滑处理
        try:
            sigma = max(1.0, min(size) / 64.0)  # 自适应平滑参数
            topography = gaussian_filter(topography, sigma=sigma)
        except Exception as e:
            self.logger.warning(f"Gaussian filtering failed: {e}")
        
        return topography
    
    def generate_time_series_topographies(self, epochs_data: np.ndarray,
                                        electrode_positions: Dict[str, Tuple[float, float]],
                                        ch_names: List[str]) -> np.ndarray:
        """为时间序列数据生成地形图序列"""
        n_epochs, n_channels, n_times = epochs_data.shape
        size = self.config.TOPO_SIZE
        
        self.logger.info(f"生成地形图序列: {n_epochs} epochs, {n_channels} channels, {n_times} time points")
        
        # 初始化输出数组
        topographies = np.zeros((n_epochs, n_times, size[0], size[1]))
        
        # 预处理电极位置信息
        valid_electrode_indices = []
        valid_positions = []
        
        for i, ch_name in enumerate(ch_names):
            if ch_name in electrode_positions:
                pos = electrode_positions[ch_name]
                if abs(pos[0]) <= 1.2 and abs(pos[1]) <= 1.2:
                    valid_electrode_indices.append(i)
                    valid_positions.append(pos)
        
        if len(valid_positions) < 3:
            self.logger.error("Not enough valid electrodes for topography generation")
            return topographies
        
        self.logger.info(f"Using {len(valid_positions)} valid electrodes for interpolation")
        
        # 生成地形图
        for epoch in range(n_epochs):
            for time_point in range(n_times):
                # 提取有效电极的数据
                valid_data = epochs_data[epoch, valid_electrode_indices, time_point]
                
                # 检查数据质量
                if np.any(np.isfinite(valid_data)) and np.std(valid_data) > 1e-10:
                    try:
                        topo = self._generate_single_topography(valid_data, valid_positions)
                        topographies[epoch, time_point] = topo
                    except Exception as e:
                        self.logger.warning(f"Failed to generate topography for epoch {epoch}, time {time_point}: {e}")
                        topographies[epoch, time_point] = np.zeros(size)
                else:
                    # 数据质量不好或全为零，使用零地形图
                    topographies[epoch, time_point] = np.zeros(size)
        
        return topographies
    
    def _generate_single_topography(self, eeg_data: np.ndarray, 
                                   positions: List[Tuple[float, float]]) -> np.ndarray:
        """生成单个地形图（内部方法，已知数据有效）"""
        size = self.config.TOPO_SIZE
        positions = np.array(positions)
        
        # 创建插值网格
        xi = np.linspace(-1.2, 1.2, size[1])
        yi = np.linspace(-1.2, 1.2, size[0])
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # 执行插值
        try:
            topography = griddata(positions, eeg_data, (xi_grid, yi_grid), 
                                method=self.config.INTERPOLATION_METHOD, 
                                fill_value=0)
            
            if np.all(np.isnan(topography)):
                topography = griddata(positions, eeg_data, (xi_grid, yi_grid), 
                                    method='linear', fill_value=0)
        except:
            topography = griddata(positions, eeg_data, (xi_grid, yi_grid), 
                                method='nearest', fill_value=0)
        
        # 处理异常值
        topography = np.nan_to_num(topography, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 应用头部掩码
        topography[~self._head_mask] = 0
        
        # 平滑处理
        sigma = max(1.0, min(size) / 64.0)
        topography = gaussian_filter(topography, sigma=sigma)
        
        return topography
    
    def normalize_topography(self, topography: np.ndarray, 
                           method: str = 'minmax') -> np.ndarray:
        """标准化地形图"""
        # 只考虑头部区域内的值
        masked_topo = topography[self._head_mask]
        
        if len(masked_topo) == 0 or np.all(masked_topo == 0):
            return topography
        
        if method == 'minmax':
            topo_min = np.min(masked_topo)
            topo_max = np.max(masked_topo)
            
            if topo_max > topo_min:
                # 只标准化头部区域
                normalized = topography.copy()
                normalized[self._head_mask] = (masked_topo - topo_min) / (topo_max - topo_min)
                return normalized
            else:
                return topography
                
        elif method == 'zscore':
            mean_val = np.mean(masked_topo)
            std_val = np.std(masked_topo)
            
            if std_val > 1e-10:
                normalized = topography.copy()
                normalized[self._head_mask] = (masked_topo - mean_val) / std_val
                return normalized
            else:
                return topography
                
        elif method == 'robust':
            # 使用中位数和四分位距进行robust标准化
            median_val = np.median(masked_topo)
            q75, q25 = np.percentile(masked_topo, [75, 25])
            iqr = q75 - q25
            
            if iqr > 1e-10:
                normalized = topography.copy()
                normalized[self._head_mask] = (masked_topo - median_val) / iqr
                return normalized
            else:
                return topography
        else:
            return topography
    
    def enhance_topography(self, topography: np.ndarray, 
                          enhancement_factor: float = 1.5) -> np.ndarray:
        """增强地形图对比度"""
        enhanced = topography.copy()
        
        # 只处理头部区域
        masked_data = enhanced[self._head_mask]
        
        if len(masked_data) > 0:
            # 使用sigmoid函数增强对比度
            mean_val = np.mean(masked_data)
            enhanced_data = mean_val + (masked_data - mean_val) * enhancement_factor
            
            # 使用tanh函数平滑截断
            enhanced_data = np.tanh(enhanced_data)
            
            enhanced[self._head_mask] = enhanced_data
        
        return enhanced
    
    def save_topography(self, topography: np.ndarray, filepath: str,
                       colormap: str = None, title: str = "") -> None:
        """保存地形图"""
        if colormap is None:
            colormap = self.config.COLORMAP
        
        plt.figure(figsize=(8, 8))
        
        # 创建地形图
        im = plt.imshow(topography, cmap=colormap, interpolation='bilinear', origin='upper')
        
        # 添加颜色条
        cbar = plt.colorbar(im, shrink=0.8)
        cbar.set_label('激活强度 (μV)', fontsize=12)
        
        # 添加头部轮廓
        center = (topography.shape[1]//2, topography.shape[0]//2)
        radius = min(topography.shape)//2 - 5
        circle = plt.Circle(center, radius, fill=False, color='black', linewidth=2)
        plt.gca().add_patch(circle)
        
        plt.title(title if title else 'EEG Topography', fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        try:
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            self.logger.info(f"Topography saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save topography to {filepath}: {e}")
        finally:
            plt.close()
    
    def get_electrode_contributions(self, topography: np.ndarray,
                                  electrode_positions: Dict[str, Tuple[float, float]],
                                  ch_names: List[str]) -> Dict[str, float]:
        """计算各电极对地形图的贡献度"""
        contributions = {}
        size = self.config.TOPO_SIZE
        
        for ch_name in ch_names:
            if ch_name in electrode_positions:
                pos = electrode_positions[ch_name]
                pixel_x, pixel_y = self.electrode_to_pixel(pos, size)
                
                # 提取电极周围区域的平均值作为贡献度
                radius = 5
                y_min = max(0, pixel_y - radius)
                y_max = min(size[0], pixel_y + radius + 1)
                x_min = max(0, pixel_x - radius)
                x_max = min(size[1], pixel_x + radius + 1)
                
                region = topography[y_min:y_max, x_min:x_max]
                contributions[ch_name] = float(np.mean(region)) if region.size > 0 else 0.0
        
        return contributions
    
    def create_difference_topography(self, topo1: np.ndarray, topo2: np.ndarray) -> np.ndarray:
        """创建差异地形图"""
        if topo1.shape != topo2.shape:
            self.logger.error("Topographies must have the same shape for difference calculation")
            return np.zeros_like(topo1)
        
        difference = topo1 - topo2
        
        # 只保留头部区域的差异
        difference[~self._head_mask] = 0
        
        return difference