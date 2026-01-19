#!/usr/bin/env python3
"""
数据生成脚本

生成唤醒词训练数据（TTS 合成 + 数据增强）
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation.tts_generator import TTSGenerator, NegativeSampleGenerator
from src.data_generation.audio_augment import augment_dataset


async def generate_tts_data(output_dir: str, negative_dir: str):
    """生成 TTS 数据"""
    print("=" * 50)
    print("Step 1: Generating TTS audio samples")
    print("=" * 50)

    # 生成正样本
    generator = TTSGenerator(output_dir)
    total_positive = await generator.generate_dataset()
    print(f"\nTotal positive samples: {total_positive}")

    # 生成负样本
    neg_generator = NegativeSampleGenerator(negative_dir)
    total_negative = await neg_generator.generate_negative_samples()
    print(f"Total negative samples: {total_negative}")

    return total_positive, total_negative


def augment_data(input_dir: str, output_dir: str, augmentations: int):
    """增强数据"""
    print("\n" + "=" * 50)
    print("Step 2: Augmenting audio samples")
    print("=" * 50)

    augment_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        augmentations_per_file=augmentations,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate wake word training data")
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
        "--augmented-dir",
        type=str,
        default="data/augmented",
        help="Output directory for augmented samples",
    )
    parser.add_argument(
        "--augmentations",
        type=int,
        default=3,
        help="Number of augmentations per file",
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Skip TTS generation (use existing data)",
    )
    parser.add_argument(
        "--skip-augment",
        action="store_true",
        help="Skip data augmentation",
    )
    args = parser.parse_args()

    # 生成 TTS 数据
    if not args.skip_tts:
        asyncio.run(generate_tts_data(args.output_dir, args.negative_dir))
    else:
        print("Skipping TTS generation")

    # 数据增强
    if not args.skip_augment:
        # 增强正样本
        augment_data(
            input_dir=args.output_dir,
            output_dir=args.augmented_dir + "/positive",
            augmentations=args.augmentations,
        )
        # 增强负样本
        augment_data(
            input_dir=args.negative_dir,
            output_dir=args.augmented_dir + "/negative",
            augmentations=args.augmentations,
        )
    else:
        print("Skipping data augmentation")

    print("\n" + "=" * 50)
    print("Data generation complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
