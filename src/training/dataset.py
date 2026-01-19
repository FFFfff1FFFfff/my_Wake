"""
Dataset Module for Wake Word Training

加载和预处理音频数据
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from scipy import signal
from torch.utils.data import Dataset, DataLoader


class AudioFeatureExtractor:
    """音频特征提取器 - 提取 Mel 频谱图"""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 40,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        fmin: int = 20,
        fmax: int = 8000,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmin = fmin
        self.fmax = fmax

        # 创建 Mel 滤波器组
        self.mel_basis = self._create_mel_filterbank()

    def _create_mel_filterbank(self) -> np.ndarray:
        """创建 Mel 滤波器组"""
        # Mel 频率转换
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        # 计算 Mel 频率点
        mel_min = hz_to_mel(self.fmin)
        mel_max = hz_to_mel(self.fmax)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # 转换为 FFT bin 索引
        bin_points = np.floor(
            (self.n_fft + 1) * hz_points / self.sample_rate
        ).astype(int)

        # 创建滤波器组
        fbank = np.zeros((self.n_mels, self.n_fft // 2 + 1))

        for i in range(self.n_mels):
            for j in range(bin_points[i], bin_points[i + 1]):
                fbank[i, j] = (j - bin_points[i]) / (bin_points[i + 1] - bin_points[i])
            for j in range(bin_points[i + 1], bin_points[i + 2]):
                fbank[i, j] = (bin_points[i + 2] - j) / (
                    bin_points[i + 2] - bin_points[i + 1]
                )

        return fbank

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        提取 Mel 频谱图特征

        Args:
            audio: 音频波形 (samples,)

        Returns:
            Mel 频谱图 (n_mels, time_frames)
        """
        # 预加重
        pre_emphasis = 0.97
        emphasized = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

        # 分帧
        num_frames = 1 + (len(emphasized) - self.win_length) // self.hop_length
        frames = np.zeros((num_frames, self.win_length))

        for i in range(num_frames):
            start = i * self.hop_length
            frames[i] = emphasized[start : start + self.win_length]

        # 加窗
        window = np.hamming(self.win_length)
        frames *= window

        # FFT
        mag_spec = np.abs(np.fft.rfft(frames, self.n_fft))

        # 应用 Mel 滤波器
        mel_spec = np.dot(mag_spec, self.mel_basis.T)

        # 对数压缩
        mel_spec = np.log(np.maximum(mel_spec, 1e-10))

        return mel_spec.T  # (n_mels, time_frames)


class WakeWordDataset(Dataset):
    """唤醒词训练数据集"""

    def __init__(
        self,
        positive_dirs: List[str],
        negative_dir: str,
        sample_rate: int = 16000,
        max_duration: float = 2.0,
        feature_extractor: Optional[AudioFeatureExtractor] = None,
    ):
        """
        Args:
            positive_dirs: 正样本目录列表（每个目录对应一个唤醒词）
            negative_dir: 负样本目录
            sample_rate: 采样率
            max_duration: 最大音频长度（秒）
            feature_extractor: 特征提取器
        """
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.feature_extractor = feature_extractor or AudioFeatureExtractor(sample_rate)

        # 加载文件路径
        self.samples: List[Tuple[Path, int]] = []  # (path, label)

        # 正样本 (label = 1)
        for pos_dir in positive_dirs:
            pos_path = Path(pos_dir)
            if pos_path.exists():
                for wav_file in pos_path.rglob("*.wav"):
                    self.samples.append((wav_file, 1))

        # 负样本 (label = 0)
        neg_path = Path(negative_dir)
        if neg_path.exists():
            for wav_file in neg_path.rglob("*.wav"):
                self.samples.append((wav_file, 0))

        # 打乱顺序
        random.shuffle(self.samples)

        print(f"Dataset loaded: {len(self.samples)} samples")
        pos_count = sum(1 for _, label in self.samples if label == 1)
        print(f"  Positive: {pos_count}, Negative: {len(self.samples) - pos_count}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_audio(self, path: Path) -> np.ndarray:
        """加载并预处理音频"""
        audio, sr = sf.read(str(path))

        # 转单声道
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # 重采样
        if sr != self.sample_rate:
            num_samples = int(len(audio) * self.sample_rate / sr)
            audio = signal.resample(audio, num_samples)

        # 填充或截断到固定长度
        if len(audio) > self.max_samples:
            # 随机裁剪
            start = random.randint(0, len(audio) - self.max_samples)
            audio = audio[start : start + self.max_samples]
        elif len(audio) < self.max_samples:
            # 填充
            padding = self.max_samples - len(audio)
            audio = np.pad(audio, (0, padding), mode="constant")

        # 归一化
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val

        return audio.astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]

        # 加载音频
        audio = self._load_audio(path)

        # 提取特征
        features = self.feature_extractor.extract(audio)

        # 转换为 tensor
        features = torch.from_numpy(features).float()

        return features, label


def create_dataloaders(
    positive_dirs: List[str],
    negative_dir: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 4,
    sample_rate: int = 16000,
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器

    Returns:
        (train_loader, val_loader)
    """
    # 创建完整数据集
    full_dataset = WakeWordDataset(
        positive_dirs=positive_dirs,
        negative_dir=negative_dir,
        sample_rate=sample_rate,
    )

    # 划分训练集和验证集
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
