"""
Audio Augmentation Module

对音频数据进行增强，提高模型的泛化能力
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy import signal
from tqdm import tqdm

try:
    import audiomentations as am

    AUDIOMENTATIONS_AVAILABLE = True
except ImportError:
    AUDIOMENTATIONS_AVAILABLE = False


class AudioAugmenter:
    """音频数据增强器"""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

        if AUDIOMENTATIONS_AVAILABLE:
            # 使用 audiomentations 库的增强管道
            self.augment_pipeline = am.Compose(
                [
                    am.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                    am.TimeStretch(min_rate=0.8, max_rate=1.2, p=0.3),
                    am.PitchShift(min_semitones=-3, max_semitones=3, p=0.3),
                    am.Shift(min_shift=-0.2, max_shift=0.2, p=0.3),
                    am.Gain(min_gain_db=-6, max_gain_db=6, p=0.3),
                    am.LowPassFilter(min_cutoff_freq=2000, max_cutoff_freq=7500, p=0.2),
                    am.HighPassFilter(min_cutoff_freq=50, max_cutoff_freq=500, p=0.2),
                ]
            )
        else:
            self.augment_pipeline = None

    def add_noise(
        self, audio: np.ndarray, noise_level: float = 0.005
    ) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.randn(len(audio)) * noise_level
        return audio + noise

    def change_speed(
        self, audio: np.ndarray, speed_factor: float
    ) -> np.ndarray:
        """改变播放速度（不改变音调）"""
        indices = np.round(np.arange(0, len(audio), speed_factor))
        indices = indices[indices < len(audio)].astype(int)
        return audio[indices]

    def change_pitch(
        self, audio: np.ndarray, semitones: float
    ) -> np.ndarray:
        """改变音调"""
        # 简单的音调变化通过重采样实现
        factor = 2 ** (semitones / 12)
        # 先拉伸/压缩
        stretched = signal.resample(audio, int(len(audio) / factor))
        # 然后调整长度回原来
        return signal.resample(stretched, len(audio))

    def add_reverb(
        self, audio: np.ndarray, decay: float = 0.3, delay_ms: float = 50
    ) -> np.ndarray:
        """添加简单的混响效果"""
        delay_samples = int(self.sample_rate * delay_ms / 1000)
        impulse_response = np.zeros(delay_samples * 5)
        for i in range(5):
            impulse_response[i * delay_samples] = decay**i
        return signal.convolve(audio, impulse_response, mode="same")

    def shift_audio(
        self, audio: np.ndarray, shift_ratio: float
    ) -> np.ndarray:
        """时间偏移"""
        shift_samples = int(len(audio) * shift_ratio)
        if shift_samples > 0:
            return np.concatenate([np.zeros(shift_samples), audio[:-shift_samples]])
        elif shift_samples < 0:
            return np.concatenate([audio[-shift_samples:], np.zeros(-shift_samples)])
        return audio

    def change_volume(
        self, audio: np.ndarray, gain_db: float
    ) -> np.ndarray:
        """改变音量"""
        gain = 10 ** (gain_db / 20)
        return audio * gain

    def augment(self, audio: np.ndarray) -> np.ndarray:
        """应用随机增强"""
        if self.augment_pipeline is not None:
            return self.augment_pipeline(audio, sample_rate=self.sample_rate)
        else:
            # 手动实现增强
            augmented = audio.copy()

            # 随机添加噪声
            if random.random() < 0.5:
                noise_level = random.uniform(0.001, 0.015)
                augmented = self.add_noise(augmented, noise_level)

            # 随机改变速度
            if random.random() < 0.3:
                speed = random.uniform(0.9, 1.1)
                augmented = self.change_speed(augmented, speed)

            # 随机音调变化
            if random.random() < 0.3:
                semitones = random.uniform(-2, 2)
                augmented = self.change_pitch(augmented, semitones)

            # 随机时间偏移
            if random.random() < 0.3:
                shift = random.uniform(-0.1, 0.1)
                augmented = self.shift_audio(augmented, shift)

            # 随机音量变化
            if random.random() < 0.3:
                gain = random.uniform(-6, 6)
                augmented = self.change_volume(augmented, gain)

            return augmented

    def augment_file(
        self,
        input_path: Path,
        output_path: Path,
        num_augmentations: int = 1,
    ) -> List[Path]:
        """对单个文件进行增强"""
        audio, sr = sf.read(str(input_path))

        if sr != self.sample_rate:
            # 重采样
            num_samples = int(len(audio) * self.sample_rate / sr)
            audio = signal.resample(audio, num_samples)

        output_paths = []
        for i in range(num_augmentations):
            augmented = self.augment(audio)

            # 归一化
            max_val = np.abs(augmented).max()
            if max_val > 0:
                augmented = augmented / max_val * 0.9

            out_path = output_path.parent / f"{output_path.stem}_aug{i}{output_path.suffix}"
            sf.write(str(out_path), augmented, self.sample_rate)
            output_paths.append(out_path)

        return output_paths


def augment_dataset(
    input_dir: str,
    output_dir: str,
    augmentations_per_file: int = 3,
    sample_rate: int = 16000,
):
    """
    对整个数据集进行增强

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        augmentations_per_file: 每个文件生成的增强版本数
        sample_rate: 采样率
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    augmenter = AudioAugmenter(sample_rate)

    # 获取所有 wav 文件
    wav_files = list(input_path.rglob("*.wav"))

    print(f"Found {len(wav_files)} files to augment")
    print(f"Will generate {len(wav_files) * augmentations_per_file} augmented files")

    for wav_file in tqdm(wav_files, desc="Augmenting"):
        # 保持相对目录结构
        rel_path = wav_file.relative_to(input_path)
        out_file = output_path / rel_path

        out_file.parent.mkdir(parents=True, exist_ok=True)

        # 复制原文件
        audio, sr = sf.read(str(wav_file))
        if sr != sample_rate:
            num_samples = int(len(audio) * sample_rate / sr)
            audio = signal.resample(audio, num_samples)
        sf.write(str(out_file), audio, sample_rate)

        # 生成增强版本
        augmenter.augment_file(wav_file, out_file, augmentations_per_file)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Augment audio dataset")
    parser.add_argument(
        "--input-dir", type=str, required=True, help="Input directory"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--augmentations", type=int, default=3, help="Augmentations per file"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="Sample rate"
    )
    args = parser.parse_args()

    augment_dataset(
        args.input_dir,
        args.output_dir,
        args.augmentations,
        args.sample_rate,
    )


if __name__ == "__main__":
    main()
