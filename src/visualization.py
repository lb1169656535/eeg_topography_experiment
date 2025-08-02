import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import cv2
from typing import Dict, List, Tuple, Optional
import os
import logging
import warnings
import platform
import matplotlib.font_manager as fm

# 抑制警告
warnings.filterwarnings('ignore', category=UserWarning)

# 检查字体支持
def check_chinese_font_support():
    """检查是否支持中文字体"""
    try:
        # 获取系统可用字体
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        system = platform.system()
        chinese_fonts = []
        
        if system == "Windows":
            chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun']
        elif system == "Darwin":  # macOS
            chinese_fonts = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti']
        else:  # Linux
            chinese_fonts = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Droid Sans Fallback']
        
        for font in chinese_fonts:
            if font in available_fonts:
                return True, font
        
        return False, None
    except:
        return False, None

# 全局字体设置
CHINESE_FONT_AVAILABLE, FONT_NAME = check_chinese_font_support()

# 多语言标签字典
LABELS = {
    'activation_intensity': '激活强度 (μV)' if CHINESE_FONT_AVAILABLE else 'Activation Intensity (μV)',
    'x_coordinate': 'X坐标 (像素)' if CHINESE_FONT_AVAILABLE else 'X Coordinate (pixels)',
    'y_coordinate': 'Y坐标 (像素)' if CHINESE_FONT_AVAILABLE else 'Y Coordinate (pixels)',
    'trajectory': '轨迹' if CHINESE_FONT_AVAILABLE else 'Trajectory',
    'frame': '帧' if CHINESE_FONT_AVAILABLE else 'Frame',
    'length': '长度' if CHINESE_FONT_AVAILABLE else 'Length',
    'intensity': '强度' if CHINESE_FONT_AVAILABLE else 'Intensity',
    'points': '点' if CHINESE_FONT_AVAILABLE else 'points',
    'total_trajectories': '轨迹总数' if CHINESE_FONT_AVAILABLE else 'Total Trajectories',
    'average_length': '平均长度' if CHINESE_FONT_AVAILABLE else 'Average Length',
    'length_range': '长度范围' if CHINESE_FONT_AVAILABLE else 'Length Range',
    'topography': '地形图' if CHINESE_FONT_AVAILABLE else 'Topography',
    'cumulative_trajectories': '累积轨迹' if CHINESE_FONT_AVAILABLE else 'Cumulative Trajectories',
    'progress': '进度' if CHINESE_FONT_AVAILABLE else 'Progress',
    'similarity_score': '相似性分数' if CHINESE_FONT_AVAILABLE else 'Similarity Score',
    'trajectory_id': '轨迹ID' if CHINESE_FONT_AVAILABLE else 'Trajectory ID',
    'clustering_results': '轨迹聚类结果' if CHINESE_FONT_AVAILABLE else 'Trajectory Clustering Results',
    'cluster': '聚类' if CHINESE_FONT_AVAILABLE else 'Cluster',
    'trajectories_count': '条轨迹' if CHINESE_FONT_AVAILABLE else 'trajectories',
    'cluster_distribution': '聚类分布' if CHINESE_FONT_AVAILABLE else 'Cluster Distribution',
    'cluster_id': '聚类ID' if CHINESE_FONT_AVAILABLE else 'Cluster ID',
    'num_trajectories': '轨迹数量' if CHINESE_FONT_AVAILABLE else 'Number of Trajectories',
    'mean': '均值' if CHINESE_FONT_AVAILABLE else 'Mean',
    'median': '中位数' if CHINESE_FONT_AVAILABLE else 'Median',
    'frequency': '频次' if CHINESE_FONT_AVAILABLE else 'Frequency',
    'value': '值' if CHINESE_FONT_AVAILABLE else 'Value',
    'feature_analysis': '轨迹特征分析' if CHINESE_FONT_AVAILABLE else 'Trajectory Feature Analysis',
    'total_distance': '总距离' if CHINESE_FONT_AVAILABLE else 'Total Distance',
    'displacement': '位移' if CHINESE_FONT_AVAILABLE else 'Displacement',
    'mean_velocity': '平均速度' if CHINESE_FONT_AVAILABLE else 'Mean Velocity',
    'tortuosity': '弯曲度' if CHINESE_FONT_AVAILABLE else 'Tortuosity',
    'straightness': '直线度' if CHINESE_FONT_AVAILABLE else 'Straightness',
    'complexity': '复杂度' if CHINESE_FONT_AVAILABLE else 'Complexity',
    'bounding_area': '覆盖面积' if CHINESE_FONT_AVAILABLE else 'Bounding Area',
    'mean_spread': '平均散布' if CHINESE_FONT_AVAILABLE else 'Mean Spread'
}

class Visualizer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 设置颜色主题
        self.colors = plt.cm.Set1(np.linspace(0, 1, 10))
        self.background_color = '#f8f9fa'
        self.grid_color = '#e9ecef'
        
        # 设置matplotlib字体
        self._setup_matplotlib_font()
        
        # 确认字体设置状态
        if CHINESE_FONT_AVAILABLE:
            self.logger.info(f"使用中文字体: {FONT_NAME}")
        else:
            self.logger.info("使用英文标签")
    
    def _setup_matplotlib_font(self):
        """设置matplotlib字体"""
        try:
            if CHINESE_FONT_AVAILABLE and FONT_NAME:
                plt.rcParams['font.sans-serif'] = [FONT_NAME] + plt.rcParams['font.sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
            else:
                # 使用安全的英文字体
                plt.rcParams['font.family'] = 'DejaVu Sans'
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
        except Exception as e:
            self.logger.warning(f"字体设置失败: {e}")
            plt.rcParams['font.family'] = 'DejaVu Sans'
    
    def setup_figure_style(self, fig, title: str = ""):
        """统一设置图形样式"""
        fig.patch.set_facecolor(self.background_color)
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
    
    def add_head_outline(self, ax, center: Tuple[float, float], radius: float, 
                        color: str = 'black', linewidth: float = 2):
        """添加头部轮廓"""
        circle = plt.Circle(center, radius, fill=False, color=color, 
                           linewidth=linewidth, alpha=0.8)
        ax.add_patch(circle)
        
        # 添加鼻子标记
        nose_x, nose_y = center[0], center[1] + radius * 0.1
        ax.plot([nose_x], [nose_y], 'k^', markersize=8, alpha=0.8)
        
        # 添加耳朵标记
        ear_y = center[1]
        left_ear_x = center[0] - radius * 1.1
        right_ear_x = center[0] + radius * 1.1
        ax.plot([left_ear_x, right_ear_x], [ear_y, ear_y], 'k-', 
                linewidth=3, alpha=0.6)
    
    def plot_topography(self, topography: np.ndarray, title: str = "", 
                       save_path: Optional[str] = None, show_colorbar: bool = True,
                       electrode_positions: Optional[Dict] = None) -> None:
        """绘制单个地形图"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            self.setup_figure_style(fig, title)
            
            # 创建地形图
            im = ax.imshow(topography, cmap=self.config.COLORMAP, 
                          interpolation='bilinear', origin='upper',
                          extent=[0, topography.shape[1], topography.shape[0], 0])
            
            # 添加颜色条
            if show_colorbar:
                cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
                cbar.set_label(LABELS['activation_intensity'], fontsize=12, fontweight='bold')
                cbar.ax.tick_params(labelsize=10)
            
            # 添加头部轮廓
            center = (topography.shape[1]//2, topography.shape[0]//2)
            radius = min(topography.shape)//2 - 5
            self.add_head_outline(ax, center, radius)
            
            # 如果提供了电极位置，标记电极
            if electrode_positions:
                self.plot_electrode_positions(ax, electrode_positions, topography.shape)
            
            # 设置坐标轴
            ax.set_xlim(0, topography.shape[1])
            ax.set_ylim(topography.shape[0], 0)
            ax.set_aspect('equal')
            ax.axis('off')
            
            # 添加标尺
            self.add_scale_bar(ax, topography.shape)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.config.DPI, bbox_inches='tight', 
                           facecolor=self.background_color)
                plt.close()
                self.logger.info(f"Topography saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Failed to plot topography: {e}")
            if 'fig' in locals():
                plt.close(fig)
    
    def plot_electrode_positions(self, ax, electrode_positions: Dict, 
                               topography_shape: Tuple[int, int]):
        """在地形图上标记电极位置"""
        for ch_name, (x, y) in electrode_positions.items():
            # 转换坐标
            pixel_x = (x + 1) * topography_shape[1] / 2
            pixel_y = (1 - y) * topography_shape[0] / 2
            
            # 绘制电极点
            ax.plot(pixel_x, pixel_y, 'wo', markersize=6, 
                   markeredgecolor='black', markeredgewidth=1)
            
            # 添加电极标签（仅显示部分以避免拥挤）
            if ch_name in ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']:
                ax.text(pixel_x, pixel_y-8, ch_name, ha='center', va='top',
                       fontsize=8, fontweight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    def add_scale_bar(self, ax, shape: Tuple[int, int]):
        """添加比例尺"""
        scale_length = shape[1] // 10
        scale_x = shape[1] * 0.05
        scale_y = shape[0] * 0.95
        
        ax.plot([scale_x, scale_x + scale_length], [scale_y, scale_y], 
               'k-', linewidth=3)
        ax.text(scale_x + scale_length/2, scale_y - 5, f'{scale_length}px',
               ha='center', va='top', fontsize=10, fontweight='bold')
    
    def plot_trajectories(self, trajectories: Dict, topography_shape: Tuple[int, int],
                         title: str = "", save_path: Optional[str] = None,
                         show_legend: bool = True, alpha: float = 0.8) -> None:
        """绘制轨迹"""
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            self.setup_figure_style(fig, title)
            
            # 创建背景
            background = np.zeros(topography_shape)
            ax.imshow(background, cmap='gray', alpha=0.2, origin='upper',
                     extent=[0, topography_shape[1], topography_shape[0], 0])
            
            # 添加网格
            ax.grid(True, alpha=0.3, color=self.grid_color, linewidth=0.5)
            
            # 绘制轨迹
            legend_elements = []
            
            for i, (traj_id, traj_data) in enumerate(trajectories.items()):
                trajectory = traj_data['trajectory']
                color = self.colors[i % len(self.colors)]
                
                # 绘制轨迹线
                line = ax.plot(trajectory[:, 1], trajectory[:, 0], 
                              color=color, linewidth=3, alpha=alpha, 
                              label=f'{LABELS["trajectory"]} {traj_id}')[0]
                
                # 标记起点
                start_point = ax.scatter(trajectory[0, 1], trajectory[0, 0], 
                                       color=color, s=150, marker='o', 
                                       edgecolors='white', linewidth=2, 
                                       zorder=5, alpha=0.9)
                
                # 标记终点
                end_point = ax.scatter(trajectory[-1, 1], trajectory[-1, 0], 
                                     color=color, s=150, marker='s', 
                                     edgecolors='white', linewidth=2, 
                                     zorder=5, alpha=0.9)
                
                # 添加方向箭头
                self.add_direction_arrows(ax, trajectory, color, alpha)
                
                # 添加轨迹信息
                trajectory_info = f"{LABELS['trajectory']} {traj_id}\n{LABELS['length']}: {len(trajectory)}{LABELS['points']}"
                
                # 安全地获取强度信息
                intensity_value = None
                for key in ['mean_intensity', 'final_intensity', 'intensity']:
                    if key in traj_data:
                        intensity_value = traj_data[key]
                        break
                
                if intensity_value is not None:
                    trajectory_info += f"\n{LABELS['intensity']}: {intensity_value:.2f}"
                
                legend_elements.append((line, trajectory_info))
            
            # 添加头部轮廓
            center = (topography_shape[1]//2, topography_shape[0]//2)
            radius = min(topography_shape)//2 - 5
            self.add_head_outline(ax, center, radius)
            
            # 设置坐标轴
            ax.set_xlim(0, topography_shape[1])
            ax.set_ylim(topography_shape[0], 0)
            ax.set_aspect('equal')
            ax.set_xlabel(LABELS['x_coordinate'], fontsize=12, fontweight='bold')
            ax.set_ylabel(LABELS['y_coordinate'], fontsize=12, fontweight='bold')
            
            # 添加图例
            if show_legend and legend_elements:
                legend_lines = [elem[0] for elem in legend_elements]
                legend_labels = [elem[1] for elem in legend_elements]
                ax.legend(legend_lines, legend_labels, bbox_to_anchor=(1.05, 1), 
                         loc='upper left', fontsize=10)
            
            # 添加统计信息
            self.add_trajectory_stats(ax, trajectories, topography_shape)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.config.DPI, bbox_inches='tight',
                           facecolor=self.background_color)
                plt.close()
                self.logger.info(f"Trajectories plot saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Failed to plot trajectories: {e}")
            if 'fig' in locals():
                plt.close(fig)
    
    def add_direction_arrows(self, ax, trajectory: np.ndarray, color, alpha: float):
        """添加轨迹方向箭头"""
        if len(trajectory) < 3:
            return
        
        # 在轨迹的几个关键点添加箭头
        arrow_positions = [len(trajectory)//4, len(trajectory)//2, 3*len(trajectory)//4]
        
        for pos in arrow_positions:
            if pos < len(trajectory) - 1:
                start = trajectory[pos]
                end = trajectory[pos + 1]
                
                dx = end[1] - start[1]
                dy = end[0] - start[0]
                
                # 只有移动足够大时才显示箭头
                if abs(dx) > 0.5 or abs(dy) > 0.5:
                    try:
                        ax.arrow(start[1], start[0], dx, dy, 
                                head_width=3, head_length=4, fc=color, ec=color,
                                alpha=alpha*0.7, length_includes_head=True)
                    except:
                        pass  # 忽略箭头绘制错误
    
    def add_trajectory_stats(self, ax, trajectories: Dict, shape: Tuple[int, int]):
        """添加轨迹统计信息"""
        stats_text = f"{LABELS['total_trajectories']}: {len(trajectories)}\n"
        
        if trajectories:
            lengths = [len(traj_data['trajectory']) for traj_data in trajectories.values()]
            stats_text += f"{LABELS['average_length']}: {np.mean(lengths):.1f}{LABELS['points']}\n"
            stats_text += f"{LABELS['length_range']}: {min(lengths)}-{max(lengths)}{LABELS['points']}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    def create_trajectory_animation(self, topographies: np.ndarray, 
                                  tracking_results: Dict,
                                  save_path: str, fps: int = None) -> None:
        """创建轨迹动画"""
        if fps is None:
            fps = self.config.FPS
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            self.setup_figure_style(fig, f"EEG {LABELS['trajectory']} Animation")
            
            n_frames = min(len(topographies), 300)  # 限制帧数以节省时间和空间
            
            def animate(frame):
                ax1.clear()
                ax2.clear()
                
                # 左侧：当前地形图
                im1 = ax1.imshow(topographies[frame], cmap=self.config.COLORMAP, 
                               interpolation='bilinear', origin='upper')
                ax1.set_title(f'{LABELS["topography"]} - {LABELS["frame"]} {frame+1}/{n_frames}', 
                             fontweight='bold')
                
                # 添加头部轮廓
                center = (topographies.shape[2]//2, topographies.shape[1]//2)
                radius = min(topographies.shape[1:])//2 - 5
                self.add_head_outline(ax1, center, radius)
                
                # 右侧：累积轨迹
                ax2.imshow(np.zeros_like(topographies[0]), cmap='gray', 
                          alpha=0.3, origin='upper')
                
                # 绘制到目前为止的轨迹
                if frame < len(tracking_results['frame_results']):
                    frame_result = tracking_results['frame_results'][frame]
                    
                    colors = plt.cm.Set1(np.linspace(0, 1, 10))
                    color_idx = 0
                    
                    for region in frame_result.get('tracked_regions', []):
                        trajectory = np.array(region.trajectory)
                        color = colors[color_idx % len(colors)]
                        
                        if len(trajectory) > 1:
                            # 绘制轨迹线
                            ax1.plot(trajectory[:, 1], trajectory[:, 0], 
                                   color=color, linewidth=2, alpha=0.8)
                            ax2.plot(trajectory[:, 1], trajectory[:, 0], 
                                   color=color, linewidth=2, alpha=0.8)
                        
                        # 标记当前位置
                        if len(trajectory) > 0:
                            current_pos = trajectory[-1]
                            ax1.scatter(current_pos[1], current_pos[0], 
                                      s=100, c=color, marker='o', 
                                      edgecolors='white', linewidth=2, zorder=5)
                            ax2.scatter(current_pos[1], current_pos[0], 
                                      s=100, c=color, marker='o', 
                                      edgecolors='white', linewidth=2, zorder=5)
                        
                        color_idx += 1
                
                # 添加头部轮廓到右侧图
                self.add_head_outline(ax2, center, radius)
                
                ax2.set_title(LABELS['cumulative_trajectories'], fontweight='bold')
                
                # 设置坐标轴
                for ax in [ax1, ax2]:
                    ax.set_xlim(0, topographies.shape[2])
                    ax.set_ylim(topographies.shape[1], 0)
                    ax.set_aspect('equal')
                    ax.axis('off')
                
                # 添加进度条
                progress = frame / (n_frames - 1)
                ax2.text(0.5, -0.05, f'{LABELS["progress"]}: {progress*100:.1f}%', 
                        transform=ax2.transAxes, ha='center', fontweight='bold')
            
            # 创建动画
            anim = animation.FuncAnimation(fig, animate, frames=n_frames, 
                                         interval=1000//fps, blit=False, repeat=True)
            
            # 保存动画
            Writer = animation.writers['pillow']
            writer = Writer(fps=fps, metadata=dict(artist='EEG Tracker'), bitrate=1800)
            anim.save(save_path, writer=writer)
            
            plt.close()
            self.logger.info(f"Animation saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save animation: {e}")
            # 尝试保存静态图像序列作为后备
            self.save_frame_sequence(topographies, tracking_results, 
                                   os.path.splitext(save_path)[0])
            if 'fig' in locals():
                plt.close(fig)
    
    def save_frame_sequence(self, topographies: np.ndarray, tracking_results: Dict,
                           base_path: str):
        """保存帧序列作为动画的后备方案"""
        frame_dir = f"{base_path}_frames"
        os.makedirs(frame_dir, exist_ok=True)
        
        n_frames_to_save = min(50, len(topographies))  # 限制帧数
        
        for frame in range(n_frames_to_save):
            try:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # 显示地形图
                ax.imshow(topographies[frame], cmap=self.config.COLORMAP, 
                         interpolation='bilinear', origin='upper')
                
                # 绘制轨迹
                if frame < len(tracking_results['frame_results']):
                    frame_result = tracking_results['frame_results'][frame]
                    
                    for i, region in enumerate(frame_result.get('tracked_regions', [])):
                        trajectory = np.array(region.trajectory)
                        color = self.colors[i % len(self.colors)]
                        
                        if len(trajectory) > 1:
                            ax.plot(trajectory[:, 1], trajectory[:, 0], 
                                   color=color, linewidth=2, alpha=0.8)
                        
                        if len(trajectory) > 0:
                            current_pos = trajectory[-1]
                            ax.scatter(current_pos[1], current_pos[0], 
                                      s=100, c=color, marker='o', 
                                      edgecolors='white', linewidth=2)
                
                # 添加头部轮廓
                center = (topographies.shape[2]//2, topographies.shape[1]//2)
                radius = min(topographies.shape[1:])//2 - 5
                self.add_head_outline(ax, center, radius)
                
                ax.set_title(f'{LABELS["frame"]} {frame+1}/{len(topographies)}', fontweight='bold')
                ax.axis('off')
                
                frame_path = os.path.join(frame_dir, f"frame_{frame:04d}.png")
                plt.savefig(frame_path, dpi=150, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                self.logger.warning(f"Failed to save frame {frame}: {e}")
                if 'fig' in locals():
                    plt.close(fig)
                continue
        
        self.logger.info(f"Frame sequence saved to {frame_dir}")
    
    def plot_similarity_matrix(self, similarity_matrix: np.ndarray,
                              labels: Optional[List[str]] = None,
                              title: str = "",
                              save_path: Optional[str] = None) -> None:
        """绘制相似性矩阵热图"""
        if not title:
            title = f"{LABELS['trajectory']} Similarity Matrix"
        
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            self.setup_figure_style(fig, title)
            
            # 创建热图
            mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
            
            sns.heatmap(similarity_matrix, 
                       annot=True, 
                       fmt='.3f',
                       cmap='RdYlBu_r', 
                       center=0.5,
                       square=True,
                       mask=mask,
                       xticklabels=labels, 
                       yticklabels=labels,
                       cbar_kws={"shrink": .8, "label": LABELS['similarity_score']},
                       ax=ax)
            
            ax.set_xlabel(LABELS['trajectory_id'], fontsize=12, fontweight='bold')
            ax.set_ylabel(LABELS['trajectory_id'], fontsize=12, fontweight='bold')
            
            # 添加统计信息
            upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            if len(upper_triangle) > 0:
                stats_text = f"Average similarity: {np.mean(upper_triangle):.3f}\n"
                stats_text += f"Range: {np.min(upper_triangle):.3f} - {np.max(upper_triangle):.3f}"
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.config.DPI, bbox_inches='tight',
                           facecolor=self.background_color)
                plt.close()
                self.logger.info(f"Similarity matrix saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Failed to plot similarity matrix: {e}")
            if 'fig' in locals():
                plt.close(fig)
    
    def plot_clustering_results(self, trajectories: Dict, clustering_results: Dict,
                               topography_shape: Tuple[int, int],
                               title: str = "",
                               save_path: Optional[str] = None) -> None:
        """绘制聚类结果"""
        if not title:
            title = LABELS['clustering_results']
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            self.setup_figure_style(fig, title)
            
            # 获取聚类信息
            labels = clustering_results['labels']
            trajectory_ids = clustering_results['trajectory_ids']
            n_clusters = clustering_results['n_clusters']
            
            # 为每个聚类分配颜色
            cluster_colors = plt.cm.Set1(np.linspace(0, 1, max(n_clusters, 2)))
            
            # 左侧：聚类轨迹图
            background = np.zeros(topography_shape)
            ax1.imshow(background, cmap='gray', alpha=0.2, origin='upper')
            
            # 记录每个聚类的轨迹数量
            cluster_counts = {}
            
            for i, traj_id in enumerate(trajectory_ids):
                if traj_id in trajectories:
                    trajectory = trajectories[traj_id]['trajectory']
                    cluster_id = labels[i]
                    color = cluster_colors[cluster_id % len(cluster_colors)]
                    
                    # 更新聚类计数
                    cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
                    
                    # 绘制轨迹
                    line = ax1.plot(trajectory[:, 1], trajectory[:, 0], 
                                   color=color, linewidth=2, alpha=0.7)[0]
                    
                    # 标记起点和终点
                    ax1.scatter(trajectory[0, 1], trajectory[0, 0], 
                               color=color, s=80, marker='o', 
                               edgecolors='black', linewidth=1, alpha=0.8)
                    ax1.scatter(trajectory[-1, 1], trajectory[-1, 0], 
                               color=color, s=80, marker='s', 
                               edgecolors='black', linewidth=1, alpha=0.8)
            
            # 添加头部轮廓
            center = (topography_shape[1]//2, topography_shape[0]//2)
            radius = min(topography_shape)//2 - 5
            self.add_head_outline(ax1, center, radius)
            
            ax1.set_xlim(0, topography_shape[1])
            ax1.set_ylim(topography_shape[0], 0)
            ax1.set_aspect('equal')
            ax1.set_title(f'Clustered Trajectories ({n_clusters} clusters)', fontweight='bold')
            ax1.axis('off')
            
            # 创建图例
            legend_elements = []
            for cluster_id, count in cluster_counts.items():
                color = cluster_colors[cluster_id % len(cluster_colors)]
                legend_elements.append(plt.Line2D([0], [0], color=color, lw=3,
                                                label=f'{LABELS["cluster"]} {cluster_id} ({count} {LABELS["trajectories_count"]})'))
            
            if legend_elements:
                ax1.legend(handles=legend_elements, loc='center left', 
                          bbox_to_anchor=(1, 0.5), fontsize=10)
            
            # 右侧：聚类统计图
            if cluster_counts:
                cluster_ids = list(cluster_counts.keys())
                counts = list(cluster_counts.values())
                
                bars = ax2.bar(cluster_ids, counts, 
                              color=[cluster_colors[i % len(cluster_colors)] for i in cluster_ids], 
                              alpha=0.7)
                
                ax2.set_xlabel(LABELS['cluster_id'], fontsize=12, fontweight='bold')
                ax2.set_ylabel(LABELS['num_trajectories'], fontsize=12, fontweight='bold')
                ax2.set_title(LABELS['cluster_distribution'], fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                # 在柱子上添加数值标签
                for bar, count in zip(bars, counts):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            str(count), ha='center', va='bottom', fontweight='bold')
                
                # 添加统计信息
                total_trajectories = sum(counts)
                avg_per_cluster = total_trajectories / len(counts) if len(counts) > 0 else 0
                stats_text = f"Total: {total_trajectories}\n"
                stats_text += f"Average per cluster: {avg_per_cluster:.1f}\n"
                stats_text += f"Largest cluster: {max(counts) if counts else 0}\n"
                stats_text += f"Smallest cluster: {min(counts) if counts else 0}"
                
                ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.config.DPI, bbox_inches='tight',
                           facecolor=self.background_color)
                plt.close()
                self.logger.info(f"Clustering results saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Failed to plot clustering results: {e}")
            if 'fig' in locals():
                plt.close(fig)
    
    def plot_trajectory_features(self, feature_data: Dict,
                               title: str = "",
                               save_path: Optional[str] = None) -> None:
        """绘制轨迹特征统计"""
        if not feature_data:
            self.logger.warning("No feature data provided for visualization")
            return
        
        if not title:
            title = LABELS['feature_analysis']
        
        try:
            # 准备数据
            features_df = []
            for traj_id, features in feature_data.items():
                row = features.copy()
                row['trajectory_id'] = traj_id
                features_df.append(row)
            
            if not features_df:
                self.logger.warning("No valid feature data")
                return
            
            try:
                import pandas as pd
                df = pd.DataFrame(features_df)
            except ImportError:
                self.logger.error("Pandas not available for feature plotting")
                return
            
            # 选择主要特征进行可视化
            feature_cols = ['total_distance', 'displacement', 'mean_velocity', 
                           'tortuosity', 'straightness', 'complexity']
            feature_cols = [col for col in feature_cols if col in df.columns]
            
            if not feature_cols:
                self.logger.warning("No valid feature columns found")
                return
            
            # 创建子图
            n_features = len(feature_cols)
            n_cols = 3
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            self.setup_figure_style(fig, title)
            
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            axes = axes.flatten()
            
            for i, feature in enumerate(feature_cols):
                if i < len(axes):
                    ax = axes[i]
                    data = df[feature].dropna()
                    
                    if len(data) > 0:
                        # 绘制直方图
                        n_bins = min(20, max(5, len(data)//3)) if len(data) > 1 else 1
                        ax.hist(data, bins=n_bins, alpha=0.7, color='skyblue', 
                               edgecolor='black', linewidth=0.5)
                        
                        # 添加统计线
                        if len(data) > 1:
                            mean_val = data.mean()
                            median_val = data.median()
                            ax.axvline(mean_val, color='red', linestyle='--', 
                                      linewidth=2, label=f'{LABELS["mean"]}: {mean_val:.3f}')
                            ax.axvline(median_val, color='orange', linestyle='--', 
                                      linewidth=2, label=f'{LABELS["median"]}: {median_val:.3f}')
                            ax.legend(fontsize=8)
                        
                        feature_name = LABELS.get(feature, feature.replace('_', ' ').title())
                        ax.set_title(feature_name, fontweight='bold')
                        ax.set_xlabel(LABELS['value'])
                        ax.set_ylabel(LABELS['frequency'])
                        ax.grid(True, alpha=0.3)
                        
                        # 添加统计信息
                        if len(data) > 1:
                            stats_text = f'N={len(data)}\nStd={data.std():.3f}'
                        else:
                            stats_text = f'N={len(data)}'
                            
                        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                               ha='right', va='top', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # 隐藏多余的子图
            for i in range(len(feature_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.config.DPI, bbox_inches='tight',
                           facecolor=self.background_color)
                plt.close()
                self.logger.info(f"Feature analysis saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Failed to plot trajectory features: {e}")
            if 'fig' in locals():
                plt.close(fig)
    
    def create_comparison_plot(self, subject_trajectories: Dict, 
                             save_path: Optional[str] = None):
        """创建被试间比较图"""
        n_subjects = len(subject_trajectories)
        if n_subjects < 2:
            self.logger.warning("Need at least 2 subjects for comparison")
            return
        
        try:
            # 创建子图网格
            n_cols = min(3, n_subjects)
            n_rows = (n_subjects + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
            self.setup_figure_style(fig, "Subject Trajectory Comparison")
            
            if n_rows == 1:
                axes = axes.reshape(1, -1) if n_subjects > 1 else [axes]
            axes = axes.flatten()
            
            # 为每个被试绘制轨迹
            for i, (subject_id, sessions) in enumerate(subject_trajectories.items()):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                
                # 收集该被试的所有轨迹
                all_trajectories = {}
                for session_id, trajectories in sessions.items():
                    for traj_id, traj_data in trajectories.items():
                        key = f"{session_id}_{traj_id}"
                        all_trajectories[key] = traj_data
                
                if all_trajectories:
                    # 假设所有轨迹有相同的地形图尺寸
                    topo_shape = (128, 128)  # 默认尺寸
                    
                    # 创建背景
                    background = np.zeros(topo_shape)
                    ax.imshow(background, cmap='gray', alpha=0.2, origin='upper')
                    
                    # 绘制轨迹
                    colors = plt.cm.Set1(np.linspace(0, 1, len(all_trajectories)))
                    
                    for j, (traj_id, traj_data) in enumerate(all_trajectories.items()):
                        trajectory = traj_data['trajectory']
                        color = colors[j % len(colors)]
                        
                        ax.plot(trajectory[:, 1], trajectory[:, 0], 
                               color=color, linewidth=2, alpha=0.7)
                        
                        # 标记起点
                        ax.scatter(trajectory[0, 1], trajectory[0, 0], 
                                 color=color, s=60, marker='o', 
                                 edgecolors='white', linewidth=1)
                    
                    # 添加头部轮廓
                    center = (topo_shape[1]//2, topo_shape[0]//2)
                    radius = min(topo_shape)//2 - 5
                    self.add_head_outline(ax, center, radius)
                    
                    ax.set_title(f'Subject {subject_id}\n({len(all_trajectories)} trajectories)', 
                               fontweight='bold')
                    ax.set_xlim(0, topo_shape[1])
                    ax.set_ylim(topo_shape[0], 0)
                    ax.set_aspect('equal')
                    ax.axis('off')
            
            # 隐藏多余的子图
            for i in range(n_subjects, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.config.DPI, bbox_inches='tight',
                           facecolor=self.background_color)
                plt.close()
                self.logger.info(f"Comparison plot saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Failed to create comparison plot: {e}")
            if 'fig' in locals():
                plt.close(fig)