import os
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import gc
from typing import List, Tuple, Dict, Optional

class EEGDataLoader:
    def __init__(self, data_root: str, config):
        self.data_root = data_root
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 设置MNE日志级别
        mne.set_log_level('WARNING')
        
    def get_subject_sessions(self, subject_id: str) -> List[str]:
        """获取指定被试的所有session"""
        subject_dir = os.path.join(self.data_root, f"sub-{subject_id}")
        if not os.path.exists(subject_dir):
            self.logger.warning(f"Subject directory not found: {subject_dir}")
            return []
        
        sessions = []
        try:
            for item in os.listdir(subject_dir):
                if item.startswith("ses-") and os.path.isdir(os.path.join(subject_dir, item)):
                    session_num = item.split("-")[1]
                    sessions.append(session_num)
        except Exception as e:
            self.logger.error(f"Error reading subject directory {subject_dir}: {e}")
            return []
        
        return sorted(sessions, key=lambda x: int(x) if x.isdigit() else 0)
    
    def find_eeg_files(self, subject_id: str, session_id: str) -> Optional[str]:
        """查找EEG文件，支持多种格式"""
        eeg_dir = os.path.join(self.data_root, f"sub-{subject_id}", f"ses-{session_id}", "eeg")
        
        if not os.path.exists(eeg_dir):
            self.logger.warning(f"EEG directory not found: {eeg_dir}")
            return None
        
        # 按优先级搜索不同格式的文件
        file_patterns = [
            f"sub-{subject_id}_ses-{session_id}_task-innerspeech_eeg.vhdr",
            f"sub-{subject_id}_ses-{session_id}_task-innerspeech_eeg.edf",
            f"sub-{subject_id}_ses-{session_id}_task-innerspeech_eeg.fif",
            f"sub-{subject_id}_ses-{session_id}_eeg.vhdr",
            f"sub-{subject_id}_ses-{session_id}_eeg.edf",
            f"sub-{subject_id}_ses-{session_id}_eeg.fif"
        ]
        
        for pattern in file_patterns:
            file_path = os.path.join(eeg_dir, pattern)
            if os.path.exists(file_path):
                return file_path
        
        # 如果找不到特定模式，列出所有EEG文件
        try:
            eeg_files = [f for f in os.listdir(eeg_dir) 
                        if f.endswith(('.vhdr', '.edf', '.fif', '.set', '.cnt'))]
            if eeg_files:
                return os.path.join(eeg_dir, eeg_files[0])
        except Exception as e:
            self.logger.error(f"Error listing EEG directory {eeg_dir}: {e}")
        
        return None
    
    def load_raw_eeg(self, subject_id: str, session_id: str) -> Optional[mne.io.Raw]:
        """加载原始EEG数据，支持多种格式"""
        eeg_file = self.find_eeg_files(subject_id, session_id)
        
        if not eeg_file:
            self.logger.warning(f"No EEG file found for subject {subject_id}, session {session_id}")
            return None
        
        try:
            # 根据文件扩展名选择加载方法
            file_ext = os.path.splitext(eeg_file)[1].lower()
            
            if file_ext == '.vhdr':
                raw = mne.io.read_raw_brainvision(eeg_file, preload=True, verbose=False)
            elif file_ext == '.edf':
                raw = mne.io.read_raw_edf(eeg_file, preload=True, verbose=False)
            elif file_ext == '.fif':
                raw = mne.io.read_raw_fif(eeg_file, preload=True, verbose=False)
            elif file_ext == '.set':
                raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
            elif file_ext == '.cnt':
                raw = mne.io.read_raw_cnt(eeg_file, preload=True, verbose=False)
            else:
                self.logger.error(f"Unsupported file format: {file_ext}")
                return None
            
            self.logger.info(f"Successfully loaded {eeg_file}")
            return raw
            
        except Exception as e:
            self.logger.error(f"Error loading EEG data from {eeg_file}: {e}")
            return None
    
    def preprocess_eeg(self, raw: mne.io.Raw) -> Optional[mne.io.Raw]:
        """预处理EEG数据"""
        try:
            # 创建副本避免修改原始数据
            raw_copy = raw.copy()
            
            # 检查并设置电极类型
            if len(raw_copy.info['chs']) > 0:
                # 自动检测EEG通道
                raw_copy.set_channel_types({ch: 'eeg' for ch in raw_copy.ch_names 
                                          if not ch.startswith(('EOG', 'ECG', 'EMG', 'TRIG', 'STIM'))})
            
            # 选择EEG通道
            raw_copy.pick_types(eeg=True, exclude='bads')
            
            if len(raw_copy.ch_names) == 0:
                self.logger.error("No EEG channels found after preprocessing")
                return None
            
            # 设置参考电极
            try:
                raw_copy.set_eeg_reference('average', projection=True, verbose=False)
                raw_copy.apply_proj(verbose=False)
            except Exception as e:
                self.logger.warning(f"Failed to set average reference: {e}")
            
            # 滤波
            try:
                raw_copy.filter(l_freq=self.config.LOW_FREQ, h_freq=self.config.HIGH_FREQ, 
                              fir_design='firwin', verbose=False)
            except Exception as e:
                self.logger.warning(f"Filtering failed: {e}")
            
            # 重采样（如果需要）
            if raw_copy.info['sfreq'] != self.config.SAMPLING_RATE:
                try:
                    raw_copy.resample(self.config.SAMPLING_RATE, verbose=False)
                except Exception as e:
                    self.logger.warning(f"Resampling failed: {e}")
            
            return raw_copy
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            return None
    
    def extract_epochs(self, raw: mne.io.Raw, epoch_length: float = None, 
                      overlap: float = 0.5, max_epochs: int = None) -> Optional[mne.Epochs]:
        """提取固定长度的epoch"""
        if epoch_length is None:
            epoch_length = self.config.TIME_WINDOW
        
        if max_epochs is None:
            max_epochs = self.config.MAX_EPOCHS_PER_SUBJECT
        
        try:
            # 创建虚拟事件
            duration = epoch_length
            interval = duration * (1 - overlap)
            
            n_samples = int(duration * raw.info['sfreq'])
            step_samples = int(interval * raw.info['sfreq'])
            
            if n_samples >= len(raw.times):
                self.logger.warning("Epoch length longer than recording, using full recording")
                n_samples = len(raw.times) - 1
            
            events = []
            event_id = 1
            
            # 限制epoch数量以节省内存
            max_start_sample = len(raw.times) - n_samples
            epoch_count = 0
            
            for start_sample in range(0, max_start_sample, step_samples):
                if epoch_count >= max_epochs:
                    break
                events.append([start_sample, 0, event_id])
                epoch_count += 1
            
            if not events:
                self.logger.error("No epochs could be created")
                return None
            
            events = np.array(events)
            
            epochs = mne.Epochs(raw, events, event_id={'epoch': event_id}, 
                               tmin=0, tmax=duration-1/raw.info['sfreq'], 
                               baseline=None, preload=True, verbose=False)
            
            # 检查epoch质量
            if len(epochs) == 0:
                self.logger.error("No valid epochs after creation")
                return None
            
            self.logger.info(f"Created {len(epochs)} epochs of {duration}s each")
            return epochs
            
        except Exception as e:
            self.logger.error(f"Epoch extraction failed: {e}")
            return None
    
    def get_electrode_positions(self, ch_names: List[str]) -> Dict[str, Tuple[float, float]]:
        """获取电极位置，改进的匹配算法"""
        positions = {}
        unmatched_channels = []
        
        for ch_name in ch_names:
            # 清理通道名称
            clean_name = ch_name.strip()
            position_found = False
            
            # 尝试多种匹配方式
            search_variants = [
                clean_name,
                clean_name.upper(),
                clean_name.lower(),
                clean_name.capitalize(),
                clean_name.replace(' ', ''),
                clean_name.replace('-', ''),
                clean_name.replace('_', '')
            ]
            
            for variant in search_variants:
                if variant in self.config.ELECTRODE_POSITIONS:
                    positions[ch_name] = self.config.ELECTRODE_POSITIONS[variant]
                    position_found = True
                    break
            
            if not position_found:
                unmatched_channels.append(ch_name)
        
        # 为未匹配的电极分配默认位置
        if unmatched_channels:
            self.logger.warning(f"Using default positions for electrodes: {unmatched_channels}")
            for i, ch_name in enumerate(unmatched_channels):
                default_pos = self.config.get_default_electrode_position(
                    ch_name, len(ch_names), len(positions) + i
                )
                positions[ch_name] = default_pos
        
        return positions
    
    def check_memory_usage(self) -> bool:
        """检查内存使用情况"""
        try:
            import psutil
            memory_usage = psutil.virtual_memory().percent
            if memory_usage > 85:  # 超过85%内存使用率
                self.logger.warning(f"High memory usage: {memory_usage:.1f}%")
                gc.collect()  # 强制垃圾回收
                return False
            return True
        except ImportError:
            return True  # 如果没有psutil，假设内存充足
    
    def load_all_subjects(self, max_subjects: Optional[int] = None) -> Dict:
        """加载所有被试数据，改进内存管理"""
        all_data = {}
        
        if max_subjects is None:
            max_subjects = self.config.MAX_SUBJECTS
        
        # 获取所有被试ID
        subject_ids = []
        try:
            for item in os.listdir(self.data_root):
                if item.startswith("sub-") and os.path.isdir(os.path.join(self.data_root, item)):
                    subject_id = item.split("-")[1]
                    subject_ids.append(subject_id)
        except Exception as e:
            self.logger.error(f"Error reading data directory {self.data_root}: {e}")
            return {}
        
        subject_ids = sorted(subject_ids, key=lambda x: int(x) if x.isdigit() else 0)
        if max_subjects:
            subject_ids = subject_ids[:max_subjects]
        
        self.logger.info(f"Found {len(subject_ids)} subjects to process")
        
        successful_loads = 0
        for subject_id in tqdm(subject_ids, desc="Loading subjects"):
            if not self.check_memory_usage():
                self.logger.warning("Memory usage too high, stopping data loading")
                break
            
            sessions = self.get_subject_sessions(subject_id)
            if not sessions:
                continue
            
            all_data[subject_id] = {}
            
            for session_id in sessions:
                try:
                    raw = self.load_raw_eeg(subject_id, session_id)
                    if raw is not None:
                        raw = self.preprocess_eeg(raw)
                        if raw is not None:
                            epochs = self.extract_epochs(raw)
                            if epochs is not None:
                                all_data[subject_id][session_id] = {
                                    'raw': raw,
                                    'epochs': epochs,
                                    'positions': self.get_electrode_positions(raw.ch_names)
                                }
                                successful_loads += 1
                                self.logger.info(f"Successfully loaded subject {subject_id}, session {session_id}")
                            else:
                                self.logger.warning(f"Failed to extract epochs for subject {subject_id}, session {session_id}")
                        else:
                            self.logger.warning(f"Failed to preprocess data for subject {subject_id}, session {session_id}")
                    else:
                        self.logger.warning(f"Failed to load raw data for subject {subject_id}, session {session_id}")
                        
                except Exception as e:
                    self.logger.error(f"Error processing subject {subject_id}, session {session_id}: {e}")
                    continue
            
            # 如果这个被试没有成功加载任何session，移除它
            if not all_data[subject_id]:
                del all_data[subject_id]
                
            # 定期清理内存
            if successful_loads % 5 == 0:
                gc.collect()
        
        self.logger.info(f"Successfully loaded data from {len(all_data)} subjects")
        return all_data