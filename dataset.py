import os
import torch
import soundfile as sf
from torch.utils.data import Dataset
import warnings
import re
import random

class DNSDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None, target_fs=48000):
        """
        Args:
            root_dir (str): Root directory of the dataset
            train (bool): True for training set, False for validation/test set
            transform (callable, optional): Optional waveform transform
            target_fs (int): Target sampling rate (reserved parameter, not actually used)
        """
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.target_fs = target_fs

        # Determine subdirectory based on train parameter
        if self.train:
            clean_dir = os.path.join(root_dir, 'generated_data/train/clean')
            noisy_dir = os.path.join(root_dir, 'generated_data/train/noisy')
        else:
            clean_dir = os.path.join(root_dir, 'generated_data/test/clean')
            noisy_dir = os.path.join(root_dir, 'generated_data/test/noisy')

        # Check if paths exist
        if not os.path.exists(clean_dir) or not os.path.exists(noisy_dir):
            raise FileNotFoundError(f"Dataset directory does not exist: {clean_dir} or {noisy_dir}")

        # Get and sort file lists
        self.clean_files = sorted([
            f for f in os.listdir(clean_dir)
            if (self.train and f.startswith("train_")) or
               (not self.train and f.startswith("test_"))
        ], key=lambda x: int(re.search(r'\d+', x).group()))

        self.noisy_files = sorted([
            f for f in os.listdir(noisy_dir)
            if (self.train and f.startswith("train_")) or
               (not self.train and f.startswith("test_"))
        ], key=lambda x: int(re.search(r'\d+', x).group()))

        # Verify file correspondence
        if len(self.clean_files) != len(self.noisy_files):
            warnings.warn("Clean and noisy audio file counts do not match!")
        else:
            # Verify indices match
            clean_indices = [int(re.search(r'\d+', f).group()) for f in self.clean_files]
            noisy_indices = [int(re.search(r'\d+', f).group()) for f in self.noisy_files]

            if clean_indices != noisy_indices:
                warnings.warn("Clean and noisy audio indices do not match!")

    def __len__(self):
        return min(len(self.clean_files), len(self.noisy_files))

    def __getitem__(self, idx):
        # Original logic: return normal noisy-clean audio pair
        if self.train:
            clean_path = os.path.join(self.root_dir, 'generated_data/train/clean', self.clean_files[idx])
            noisy_path = os.path.join(self.root_dir, 'generated_data/train/noisy', self.noisy_files[idx])
        else:
            clean_path = os.path.join(self.root_dir, 'generated_data/test/clean', self.clean_files[idx])
            noisy_path = os.path.join(self.root_dir, 'generated_data/test/noisy', self.noisy_files[idx])

        # 检查文件是否存在
        if not os.path.exists(clean_path):
            raise FileNotFoundError(f"文件不存在: {clean_path}")
        if not os.path.exists(noisy_path):
            raise FileNotFoundError(f"文件不存在: {noisy_path}")

        try:
            # 只使用soundfile加载音频
            clean_waveform, sr = sf.read(clean_path)
            noisy_waveform, sr = sf.read(noisy_path)
            # 转换为torch张量并添加通道维度
            clean_waveform = torch.tensor(clean_waveform).unsqueeze(0)
            noisy_waveform = torch.tensor(noisy_waveform).unsqueeze(0)
        except Exception as e:
            raise RuntimeError(f"加载音频失败: {clean_path} 或 {noisy_path}. 错误: {str(e)}")

        # 统一音频长度
        min_len = min(clean_waveform.shape[1], noisy_waveform.shape[1])
        clean_waveform = clean_waveform[:, :min_len]
        noisy_waveform = noisy_waveform[:, :min_len]

        if self.transform:
            clean_waveform = self.transform(clean_waveform)
            noisy_waveform = self.transform(noisy_waveform)

        return noisy_waveform, clean_waveform


if __name__ == "__main__":
    # 测试训练集
    train_dataset = DNSDataset(root_dir="_datasets_fullband", train=True)
    print(f"训练集大小: {len(train_dataset)}")
    noisy_train, clean_train = train_dataset[0]
    print(f"训练集噪声音频形状: {noisy_train.shape}, 干净音频形状: {clean_train.shape}")
    
    # 测试验证集
    val_dataset = DNSDataset(root_dir="_datasets_fullband", train=False)
    print(f"验证集大小: {len(val_dataset)}")
    noisy_val, clean_val = val_dataset[0]
    print(f"验证集噪声音频形状: {noisy_val.shape}, 干净音频形状: {clean_val.shape}")