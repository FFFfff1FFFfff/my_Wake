"""
TTS Data Generator for Wake Word Training

使用多种 TTS 引擎生成唤醒词的合成音频数据
"""

import asyncio
import os
import random
from pathlib import Path
from typing import List, Optional

import edge_tts
import numpy as np
import soundfile as sf
from scipy import signal
from tqdm import tqdm


# 唤醒词列表
WAKE_WORDS = {
    "hi_iroi": ["hi iroi", "hi, iroi", "hi iroi!"],
    "hey_iroi": ["hey iroi", "hey, iroi", "hey iroi!"],
    "hello_iroi": ["hello iroi", "hello, iroi", "hello iroi!"],
}

# Edge TTS 可用的英文声音
EDGE_VOICES = [
    "en-US-AriaNeural",
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-ChristopherNeural",
    "en-US-EricNeural",
    "en-US-MichelleNeural",
    "en-US-RogerNeural",
    "en-US-SteffanNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-GB-LibbyNeural",
    "en-AU-NatashaNeural",
    "en-AU-WilliamNeural",
    "en-CA-ClaraNeural",
    "en-CA-LiamNeural",
    "en-IN-NeerjaNeural",
    "en-IN-PrabhatNeural",
]

# 语速变化范围
RATE_VARIATIONS = ["-20%", "-10%", "+0%", "+10%", "+20%"]

# 音调变化范围
PITCH_VARIATIONS = ["-10Hz", "-5Hz", "+0Hz", "+5Hz", "+10Hz"]


class TTSGenerator:
    """使用 Edge TTS 生成唤醒词音频"""

    def __init__(self, output_dir: str, sample_rate: int = 16000):
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def generate_single(
        self,
        text: str,
        voice: str,
        output_path: Path,
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> bool:
        """生成单个音频文件"""
        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            temp_path = output_path.with_suffix(".mp3")
            await communicate.save(str(temp_path))

            # 转换为 16kHz wav
            self._convert_to_wav(temp_path, output_path)
            temp_path.unlink()  # 删除临时 mp3 文件
            return True
        except Exception as e:
            print(f"Error generating {output_path}: {e}")
            return False

    def _convert_to_wav(self, input_path: Path, output_path: Path):
        """将音频转换为 16kHz 单声道 wav"""
        import subprocess

        # 使用 ffmpeg 转换 (如果可用)
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(input_path),
                    "-ar",
                    str(self.sample_rate),
                    "-ac",
                    "1",
                    str(output_path),
                ],
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # 如果 ffmpeg 不可用，使用 scipy 处理
            data, sr = sf.read(str(input_path))
            if len(data.shape) > 1:
                data = data.mean(axis=1)  # 转单声道
            if sr != self.sample_rate:
                # 重采样
                num_samples = int(len(data) * self.sample_rate / sr)
                data = signal.resample(data, num_samples)
            sf.write(str(output_path), data, self.sample_rate)

    async def generate_dataset(
        self,
        wake_words: dict = WAKE_WORDS,
        samples_per_combination: int = 1,
        voices: List[str] = None,
        rates: List[str] = None,
        pitches: List[str] = None,
    ) -> int:
        """
        生成完整的唤醒词数据集

        Args:
            wake_words: 唤醒词字典 {类别: [变体列表]}
            samples_per_combination: 每个组合生成的样本数
            voices: 要使用的声音列表
            rates: 语速变化列表
            pitches: 音调变化列表

        Returns:
            生成的文件总数
        """
        voices = voices or EDGE_VOICES
        rates = rates or RATE_VARIATIONS
        pitches = pitches or PITCH_VARIATIONS

        total_generated = 0

        for category, texts in wake_words.items():
            category_dir = self.output_dir / category
            category_dir.mkdir(exist_ok=True)

            # 计算总任务数用于进度条
            total_tasks = (
                len(texts)
                * len(voices)
                * len(rates)
                * len(pitches)
                * samples_per_combination
            )

            print(f"\nGenerating '{category}' samples...")
            pbar = tqdm(total=total_tasks, desc=category)

            file_idx = 0
            for text in texts:
                for voice in voices:
                    for rate in rates:
                        for pitch in pitches:
                            for _ in range(samples_per_combination):
                                output_path = category_dir / f"{file_idx:05d}.wav"
                                success = await self.generate_single(
                                    text=text,
                                    voice=voice,
                                    output_path=output_path,
                                    rate=rate,
                                    pitch=pitch,
                                )
                                if success:
                                    total_generated += 1
                                    file_idx += 1
                                pbar.update(1)

            pbar.close()
            print(f"Generated {file_idx} samples for '{category}'")

        return total_generated


class NegativeSampleGenerator:
    """生成负样本（非唤醒词的音频）"""

    # 负样本文本 - 相似但不同的短语
    NEGATIVE_TEXTS = [
        # 相似的问候语
        "hi there",
        "hey there",
        "hello there",
        "hi everyone",
        "hey you",
        "hello world",
        # 相似发音
        "hi Roy",
        "hey Roy",
        "hello Roy",
        "hi Elroy",
        "hi Troy",
        # 常见短语
        "hi",
        "hey",
        "hello",
        "good morning",
        "good evening",
        "how are you",
        "what's up",
        "excuse me",
        # 其他随机短语
        "the weather is nice",
        "can you help me",
        "let me think",
        "I don't know",
        "yes please",
        "no thanks",
        "maybe later",
        "see you soon",
    ]

    def __init__(self, output_dir: str, sample_rate: int = 16000):
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tts_generator = TTSGenerator(output_dir, sample_rate)

    async def generate_negative_samples(
        self, samples_per_text: int = 10, voices: List[str] = None
    ) -> int:
        """生成负样本"""
        voices = voices or EDGE_VOICES[:5]  # 使用部分声音
        rates = ["-10%", "+0%", "+10%"]
        pitches = ["-5Hz", "+0Hz", "+5Hz"]

        total_generated = 0
        file_idx = 0

        print("\nGenerating negative samples...")
        total_tasks = (
            len(self.NEGATIVE_TEXTS) * len(voices) * len(rates) * len(pitches)
        )
        pbar = tqdm(total=total_tasks)

        for text in self.NEGATIVE_TEXTS:
            for voice in voices:
                for rate in rates:
                    for pitch in pitches:
                        output_path = self.output_dir / f"neg_{file_idx:05d}.wav"
                        success = await self.tts_generator.generate_single(
                            text=text,
                            voice=voice,
                            output_path=output_path,
                            rate=rate,
                            pitch=pitch,
                        )
                        if success:
                            total_generated += 1
                            file_idx += 1
                        pbar.update(1)

        pbar.close()
        print(f"Generated {total_generated} negative samples")
        return total_generated


async def main():
    """主函数：生成完整数据集"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate TTS wake word dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/positive",
        help="Output directory for positive samples",
    )
    parser.add_argument(
        "--negative-dir",
        type=str,
        default="data/negative",
        help="Output directory for negative samples",
    )
    parser.add_argument(
        "--positive-only", action="store_true", help="Only generate positive samples"
    )
    parser.add_argument(
        "--negative-only", action="store_true", help="Only generate negative samples"
    )
    args = parser.parse_args()

    # 生成正样本
    if not args.negative_only:
        generator = TTSGenerator(args.output_dir)
        total = await generator.generate_dataset()
        print(f"\nTotal positive samples generated: {total}")

    # 生成负样本
    if not args.positive_only:
        neg_generator = NegativeSampleGenerator(args.negative_dir)
        total_neg = await neg_generator.generate_negative_samples()
        print(f"\nTotal negative samples generated: {total_neg}")


if __name__ == "__main__":
    asyncio.run(main())
