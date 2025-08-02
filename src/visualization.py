import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import cv2
from typing import Dict, List, Tuple, Optional
import os
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Set up matplotlib for English only
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']

class Visualizer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set color theme
        self.colors = plt.cm.Set1(np.linspace(0, 1, 10))
        self.background_color = '#f8f9fa'
        self.grid_color = '#e9ecef'
        
        self.logger.info("Visualizer initialized with English labels")
    
    def setup_figure_style(self, fig, title: str = ""):
        """Set unified figure style"""
        fig.patch.set_facecolor(self.background_color)
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
    
    def add_head_outline(self, ax, center: Tuple[float, float], radius: float, 
                        color: str = 'black', linewidth: float = 2):
        """Add head outline"""
        circle = plt.Circle(center, radius, fill=False, color=color, 
                           linewidth=linewidth, alpha=0.8)
        ax.add_patch(circle)
        
        # Add nose marker
        nose_x, nose_y = center[0], center[1] + radius * 0.1
        ax.plot([nose_x], [nose_y], 'k^', markersize=8, alpha=0.8)
        
        # Add ear markers
        ear_y = center[1]
        left_ear_x = center[0] - radius * 1.1
        right_ear_x = center[0] + radius * 1.1
        ax.plot([left_ear_x, right_ear_x], [ear_y, ear_y], 'k-', 
                linewidth=3, alpha=0.6)
    
    def plot_topography(self, topography: np.ndarray, title: str = "", 
                       save_path: Optional[str] = None, show_colorbar: bool = True,
                       electrode_positions: Optional[Dict] = None) -> None:
        """Plot single topography"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            self.setup_figure_style(fig, title)
            
            # Create topography
            im = ax.imshow(topography, cmap=self.config.COLORMAP, 
                          interpolation='bilinear', origin='upper',
                          extent=[0, topography.shape[1], topography.shape[0], 0])
            
            # Add colorbar
            if show_colorbar:
                cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
                cbar.set_label('Activation Intensity (μV)', fontsize=12, fontweight='bold')
                cbar.ax.tick_params(labelsize=10)
            
            # Add head outline
            center = (topography.shape[1]//2, topography.shape[0]//2)
            radius = min(topography.shape)//2 - 5
            self.add_head_outline(ax, center, radius)
            
            # Mark electrodes if provided
            if electrode_positions:
                self.plot_electrode_positions(ax, electrode_positions, topography.shape)
            
            # Set axes
            ax.set_xlim(0, topography.shape[1])
            ax.set_ylim(topography.shape[0], 0)
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Add scale bar
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
        """Mark electrode positions on topography"""
        for ch_name, (x, y) in electrode_positions.items():
            # Convert coordinates
            pixel_x = (x + 1) * topography_shape[1] / 2
            pixel_y = (1 - y) * topography_shape[0] / 2
            
            # Draw electrode point
            ax.plot(pixel_x, pixel_y, 'wo', markersize=6, 
                   markeredgecolor='black', markeredgewidth=1)
            
            # Add electrode label (only for key electrodes to avoid crowding)
            if ch_name in ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']:
                ax.text(pixel_x, pixel_y-8, ch_name, ha='center', va='top',
                       fontsize=8, fontweight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    def add_scale_bar(self, ax, shape: Tuple[int, int]):
        """Add scale bar"""
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
        """Plot trajectories"""
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            self.setup_figure_style(fig, title)
            
            # Create background
            background = np.zeros(topography_shape)
            ax.imshow(background, cmap='gray', alpha=0.2, origin='upper',
                     extent=[0, topography_shape[1], topography_shape[0], 0])
            
            # Add grid
            ax.grid(True, alpha=0.3, color=self.grid_color, linewidth=0.5)
            
            # Plot trajectories
            legend_elements = []
            
            for i, (traj_id, traj_data) in enumerate(trajectories.items()):
                trajectory = traj_data['trajectory']
                color = self.colors[i % len(self.colors)]
                
                # Plot trajectory line
                line = ax.plot(trajectory[:, 1], trajectory[:, 0], 
                              color=color, linewidth=3, alpha=alpha, 
                              label=f'Trajectory {traj_id}')[0]
                
                # Mark start point
                start_point = ax.scatter(trajectory[0, 1], trajectory[0, 0], 
                                       color=color, s=150, marker='o', 
                                       edgecolors='white', linewidth=2, 
                                       zorder=5, alpha=0.9)
                
                # Mark end point
                end_point = ax.scatter(trajectory[-1, 1], trajectory[-1, 0], 
                                     color=color, s=150, marker='s', 
                                     edgecolors='white', linewidth=2, 
                                     zorder=5, alpha=0.9)
                
                # Add direction arrows
                self.add_direction_arrows(ax, trajectory, color, alpha)
                
                # Add trajectory info
                trajectory_info = f"Trajectory {traj_id}\nLength: {len(trajectory)} points"
                
                # Safely get intensity information
                intensity_value = None
                for key in ['mean_intensity', 'final_intensity', 'intensity']:
                    if key in traj_data:
                        intensity_value = traj_data[key]
                        break
                
                if intensity_value is not None:
                    trajectory_info += f"\nIntensity: {intensity_value:.2f}"
                
                legend_elements.append((line, trajectory_info))
            
            # Add head outline
            center = (topography_shape[1]//2, topography_shape[0]//2)
            radius = min(topography_shape)//2 - 5
            self.add_head_outline(ax, center, radius)
            
            # Set axes
            ax.set_xlim(0, topography_shape[1])
            ax.set_ylim(topography_shape[0], 0)
            ax.set_aspect('equal')
            ax.set_xlabel('X Coordinate (pixels)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Y Coordinate (pixels)', fontsize=12, fontweight='bold')
            
            # Add legend
            if show_legend and legend_elements:
                legend_lines = [elem[0] for elem in legend_elements]
                legend_labels = [elem[1] for elem in legend_elements]
                ax.legend(legend_lines, legend_labels, bbox_to_anchor=(1.05, 1), 
                         loc='upper left', fontsize=10)
            
            # Add statistics
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
        """Add trajectory direction arrows"""
        if len(trajectory) < 3:
            return
        
        # Add arrows at key points in trajectory
        arrow_positions = [len(trajectory)//4, len(trajectory)//2, 3*len(trajectory)//4]
        
        for pos in arrow_positions:
            if pos < len(trajectory) - 1:
                start = trajectory[pos]
                end = trajectory[pos + 1]
                
                dx = end[1] - start[1]
                dy = end[0] - start[0]
                
                # Only show arrow if movement is significant
                if abs(dx) > 0.5 or abs(dy) > 0.5:
                    try:
                        ax.arrow(start[1], start[0], dx, dy, 
                                head_width=3, head_length=4, fc=color, ec=color,
                                alpha=alpha*0.7, length_includes_head=True)
                    except:
                        pass  # Ignore arrow drawing errors
    
    def add_trajectory_stats(self, ax, trajectories: Dict, shape: Tuple[int, int]):
        """Add trajectory statistics"""
        stats_text = f"Total Trajectories: {len(trajectories)}\n"
        
        if trajectories:
            lengths = [len(traj_data['trajectory']) for traj_data in trajectories.values()]
            stats_text += f"Average Length: {np.mean(lengths):.1f} points\n"
            stats_text += f"Length Range: {min(lengths)}-{max(lengths)} points"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    def create_algorithm_comparison_plot(self, algorithm_results: Dict, save_path: str):
        """Create comprehensive algorithm comparison visualization"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('EEG Trajectory Tracking Algorithm Comparison', fontsize=16, fontweight='bold')
            
            algorithms = list(algorithm_results.keys())
            colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
            
            # 1. Trajectory Count Comparison
            ax = axes[0, 0]
            trajectory_counts = []
            for alg in algorithms:
                count = algorithm_results[alg].get('avg_trajectories', 0)
                trajectory_counts.append(count)
            
            bars = ax.bar(algorithms, trajectory_counts, color=colors, alpha=0.7)
            ax.set_title('Average Trajectory Count', fontweight='bold')
            ax.set_ylabel('Number of Trajectories')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, count in zip(bars, trajectory_counts):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{count:.1f}', ha='center', va='bottom')
            
            # 2. Computation Time Comparison
            ax = axes[0, 1]
            comp_times = []
            for alg in algorithms:
                time = algorithm_results[alg].get('avg_time', 0)
                comp_times.append(time)
            
            bars = ax.bar(algorithms, comp_times, color=colors, alpha=0.7)
            ax.set_title('Average Computation Time', fontweight='bold')
            ax.set_ylabel('Time (seconds)')
            ax.tick_params(axis='x', rotation=45)
            
            for bar, time in zip(bars, comp_times):
                if time > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{time:.3f}s', ha='center', va='bottom', fontsize=9)
            
            # 3. Trajectory Quality Comparison
            ax = axes[0, 2]
            qualities = []
            for alg in algorithms:
                quality = algorithm_results[alg].get('avg_quality', 0)
                qualities.append(quality)
            
            bars = ax.bar(algorithms, qualities, color=colors, alpha=0.7)
            ax.set_title('Average Trajectory Quality', fontweight='bold')
            ax.set_ylabel('Quality Score')
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylim(0, 1)
            
            for bar, quality in zip(bars, qualities):
                if quality > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{quality:.3f}', ha='center', va='bottom')
            
            # 4. Trajectory Length Comparison
            ax = axes[1, 0]
            lengths = []
            for alg in algorithms:
                length = algorithm_results[alg].get('avg_length', 0)
                lengths.append(length)
            
            bars = ax.bar(algorithms, lengths, color=colors, alpha=0.7)
            ax.set_title('Average Trajectory Length', fontweight='bold')
            ax.set_ylabel('Length (frames)')
            ax.tick_params(axis='x', rotation=45)
            
            for bar, length in zip(bars, lengths):
                if length > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{length:.1f}', ha='center', va='bottom')
            
            # 5. Performance Efficiency (Trajectories per Second)
            ax = axes[1, 1]
            efficiencies = []
            for alg in algorithms:
                traj_count = algorithm_results[alg].get('avg_trajectories', 0)
                time = algorithm_results[alg].get('avg_time', 1e-6)  # Avoid division by zero
                efficiency = traj_count / time if time > 0 else 0
                efficiencies.append(efficiency)
            
            bars = ax.bar(algorithms, efficiencies, color=colors, alpha=0.7)
            ax.set_title('Processing Efficiency', fontweight='bold')
            ax.set_ylabel('Trajectories per Second')
            ax.tick_params(axis='x', rotation=45)
            
            for bar, eff in zip(bars, efficiencies):
                if eff > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{eff:.1f}', ha='center', va='bottom')
            
            # 6. Overall Performance Score
            ax = axes[1, 2]
            performance_scores = []
            for alg in algorithms:
                # Calculate composite performance score
                traj_score = min(1.0, algorithm_results[alg].get('avg_trajectories', 0) / 5.0)
                quality_score = algorithm_results[alg].get('avg_quality', 0)
                time_penalty = max(0, 1.0 - algorithm_results[alg].get('avg_time', 0) / 0.5)
                
                overall_score = (traj_score * 0.4 + quality_score * 0.4 + time_penalty * 0.2)
                performance_scores.append(overall_score)
            
            bars = ax.bar(algorithms, performance_scores, color=colors, alpha=0.7)
            ax.set_title('Overall Performance Score', fontweight='bold')
            ax.set_ylabel('Composite Score')
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylim(0, 1)
            
            for bar, score in zip(bars, performance_scores):
                if score > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{score:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Algorithm comparison plot saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create algorithm comparison plot: {e}")
            if 'fig' in locals():
                plt.close(fig)
    
    def create_performance_radar_chart(self, algorithm_results: Dict, save_path: str):
        """Create radar chart for algorithm performance comparison"""
        try:
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # Performance metrics
            metrics = ['Trajectory Count', 'Quality Score', 'Speed', 'Efficiency', 'Stability']
            algorithms = list(algorithm_results.keys())
            colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
            
            # Normalize metrics to 0-1 scale
            max_values = {}
            for metric in metrics:
                if metric == 'Trajectory Count':
                    max_values[metric] = max([alg.get('avg_trajectories', 0) for alg in algorithm_results.values()])
                elif metric == 'Quality Score':
                    max_values[metric] = 1.0
                elif metric == 'Speed':
                    min_time = min([alg.get('avg_time', 1) for alg in algorithm_results.values() if alg.get('avg_time', 1) > 0])
                    max_values[metric] = min_time  # Lower time is better
                elif metric == 'Efficiency':
                    max_values[metric] = max([alg.get('avg_trajectories', 0) / max(alg.get('avg_time', 1), 1e-6) 
                                            for alg in algorithm_results.values()])
                elif metric == 'Stability':
                    max_values[metric] = 1.0
            
            # Plot each algorithm
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            for i, (alg_name, alg_data) in enumerate(algorithm_results.items()):
                values = []
                
                # Calculate normalized values
                traj_count = alg_data.get('avg_trajectories', 0)
                values.append(traj_count / max(max_values['Trajectory Count'], 1))
                
                quality = alg_data.get('avg_quality', 0)
                values.append(quality)
                
                time = alg_data.get('avg_time', 1)
                speed_score = max_values['Speed'] / max(time, 1e-6) if time > 0 else 0
                values.append(min(1.0, speed_score))
                
                efficiency = traj_count / max(time, 1e-6)
                values.append(efficiency / max(max_values['Efficiency'], 1))
                
                stability = 1.0 - min(1.0, alg_data.get('computation_time_std', 0) / max(time, 1e-6))
                values.append(max(0, stability))
                
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, label=alg_name, 
                       color=colors[i % len(colors)])
                ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
            
            # Customize the chart
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1)
            ax.set_title('Algorithm Performance Radar Chart', size=16, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Performance radar chart saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create radar chart: {e}")
            if 'fig' in locals():
                plt.close(fig)
    
    def create_detailed_comparison_table(self, algorithm_results: Dict, save_path: str):
        """Create detailed comparison table visualization"""
        try:
            # Prepare data for table
            data = []
            headers = ['Algorithm', 'Avg Trajectories', 'Avg Quality', 'Avg Time (s)', 
                      'Efficiency', 'Best For']
            
            best_for = {
                'greedy': 'Real-time processing',
                'hungarian': 'High precision tasks',
                'kalman': 'Predictable motion',
                'overlap': 'Shape-stable objects',
                'hybrid': 'Complex scenarios'
            }
            
            for alg_name, alg_data in algorithm_results.items():
                row = [
                    alg_name.capitalize(),
                    f"{alg_data.get('avg_trajectories', 0):.2f}",
                    f"{alg_data.get('avg_quality', 0):.3f}",
                    f"{alg_data.get('avg_time', 0):.4f}",
                    f"{alg_data.get('avg_trajectories', 0) / max(alg_data.get('avg_time', 1), 1e-6):.1f}",
                    best_for.get(alg_name, 'General purpose')
                ]
                data.append(row)
            
            # Create table plot
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.axis('tight')
            ax.axis('off')
            
            table = ax.table(cellText=data, colLabels=headers, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Style the table
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Color rows alternately
            for i in range(1, len(data) + 1):
                for j in range(len(headers)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f1f1f2')
                    else:
                        table[(i, j)].set_facecolor('white')
            
            plt.title('Algorithm Performance Comparison Table', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Comparison table saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create comparison table: {e}")
            if 'fig' in locals():
                plt.close(fig)
    
    def create_trajectory_animation(self, topographies: np.ndarray, 
                                  tracking_results: Dict,
                                  save_path: str, fps: int = None) -> None:
        """Create trajectory animation (simplified version)"""
        if fps is None:
            fps = self.config.FPS
        
        try:
            # Save frame sequence instead of animation for better compatibility
            frame_dir = f"{os.path.splitext(save_path)[0]}_frames"
            os.makedirs(frame_dir, exist_ok=True)
            
            n_frames = min(len(topographies), self.config.MAX_SAVE_FRAMES)
            
            for frame in range(n_frames):
                try:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Display topography
                    ax.imshow(topographies[frame], cmap=self.config.COLORMAP, 
                             interpolation='bilinear', origin='upper')
                    
                    # Draw trajectories up to current frame
                    if frame < len(tracking_results.get('frame_results', [])):
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
                    
                    # Add head outline
                    center = (topographies.shape[2]//2, topographies.shape[1]//2)
                    radius = min(topographies.shape[1:])//2 - 5
                    self.add_head_outline(ax, center, radius)
                    
                    ax.set_title(f'Frame {frame+1}/{n_frames}', fontweight='bold')
                    ax.axis('off')
                    
                    frame_path = os.path.join(frame_dir, f"frame_{frame:04d}.png")
                    plt.savefig(frame_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                except Exception as e:
                    self.logger.warning(f"Failed to save frame {frame}: {e}")
                    if 'fig' in locals():
                        plt.close(fig)
                    continue
            
            self.logger.info(f"Animation frames saved to {frame_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to create animation: {e}")
    
    def create_summary_visualization(self, all_results: Dict, save_path: str):
        """Create comprehensive summary visualization"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('EEG Trajectory Analysis Summary', fontsize=16, fontweight='bold')
            
            # Extract summary statistics
            subject_counts = {}
            algorithm_performance = {}
            
            for subject_id, sessions in all_results.items():
                for session_id, session_data in sessions.items():
                    for algorithm_name, alg_data in session_data.items():
                        if algorithm_name not in algorithm_performance:
                            algorithm_performance[algorithm_name] = {
                                'trajectory_counts': [],
                                'computation_times': [],
                                'qualities': []
                            }
                        
                        algorithm_performance[algorithm_name]['trajectory_counts'].append(
                            alg_data.get('total_trajectories', 0))
                        algorithm_performance[algorithm_name]['computation_times'].append(
                            alg_data.get('total_computation_time', 0))
                        
                        # Calculate average quality
                        qualities = []
                        for traj_data in alg_data.get('trajectories', {}).values():
                            qualities.append(traj_data.get('quality_score', 0))
                        
                        if qualities:
                            algorithm_performance[algorithm_name]['qualities'].append(np.mean(qualities))
            
            # 1. Algorithm Performance Summary
            ax = axes[0, 0]
            algorithms = list(algorithm_performance.keys())
            avg_trajectories = [np.mean(algorithm_performance[alg]['trajectory_counts']) 
                              for alg in algorithms]
            
            bars = ax.bar(algorithms, avg_trajectories, alpha=0.7)
            ax.set_title('Average Trajectories per Algorithm')
            ax.set_ylabel('Number of Trajectories')
            ax.tick_params(axis='x', rotation=45)
            
            # 2. Processing Time Comparison
            ax = axes[0, 1]
            avg_times = [np.mean(algorithm_performance[alg]['computation_times']) 
                        for alg in algorithms]
            
            bars = ax.bar(algorithms, avg_times, alpha=0.7, color='orange')
            ax.set_title('Average Processing Time')
            ax.set_ylabel('Time (seconds)')
            ax.tick_params(axis='x', rotation=45)
            
            # 3. Quality Distribution
            ax = axes[1, 0]
            all_qualities = []
            labels = []
            
            for alg in algorithms:
                qualities = algorithm_performance[alg]['qualities']
                if qualities:
                    all_qualities.extend(qualities)
                    labels.extend([alg] * len(qualities))
            
            if all_qualities:
                unique_algorithms = list(set(labels))
                quality_data = [algorithm_performance[alg]['qualities'] for alg in unique_algorithms]
                
                ax.boxplot(quality_data, labels=unique_algorithms)
                ax.set_title('Quality Score Distribution')
                ax.set_ylabel('Quality Score')
                ax.tick_params(axis='x', rotation=45)
            
            # 4. Summary Statistics
            ax = axes[1, 1]
            ax.axis('off')
            
            # Create summary text
            total_subjects = len(all_results)
            total_sessions = sum(len(sessions) for sessions in all_results.values())
            total_algorithms = len(algorithms)
            
            summary_text = f"""
Experiment Summary:
• Total Subjects: {total_subjects}
• Total Sessions: {total_sessions}  
• Algorithms Compared: {total_algorithms}
• Algorithms: {', '.join(algorithms)}

Best Performing Algorithm:
• Most Trajectories: {algorithms[np.argmax(avg_trajectories)]}
• Fastest Processing: {algorithms[np.argmin(avg_times)]}
"""
            
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Summary visualization saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create summary visualization: {e}")
            if 'fig' in locals():
                plt.close(fig)