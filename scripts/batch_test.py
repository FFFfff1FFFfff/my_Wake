#!/usr/bin/env python3
"""
批量测试脚本

测试目录结构:
tests/
├── positive/    # 正样本（应该检测到）
│   ├── hi_iroi_1.wav
│   ├── hey_iroi_1.wav
│   └── ...
└── negative/    # 负样本（不应该检测到）
    ├── hi_there.wav
    ├── random_speech.wav
    └── ...

用法:
    python scripts/batch_test.py --model models/wake_word_model.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.wake_detector import WakeWordDetector


def load_audio(file_path: str, sample_rate: int = 16000) -> np.ndarray:
    """加载音频文件"""
    audio, sr = sf.read(file_path)

    # 转单声道
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # 重采样
    if sr != sample_rate:
        num_samples = int(len(audio) * sample_rate / sr)
        audio = signal.resample(audio, num_samples)

    return audio.astype(np.float32)


def test_directory(
    detector: WakeWordDetector,
    directory: Path,
    expected_result: bool,
    threshold: float,
) -> dict:
    """测试一个目录下的所有音频"""
    results = []

    # 支持的音频格式
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    audio_files = [
        f for f in directory.iterdir()
        if f.suffix.lower() in audio_extensions
    ]

    if not audio_files:
        print(f"  No audio files found in {directory}")
        return {"total": 0, "correct": 0, "files": []}

    for audio_file in sorted(audio_files):
        try:
            audio = load_audio(str(audio_file))
            confidence = detector.detect(audio)
            detected = confidence > threshold
            correct = detected == expected_result

            results.append({
                "file": audio_file.name,
                "confidence": confidence,
                "detected": detected,
                "expected": expected_result,
                "correct": correct,
            })

            # 显示结果
            status = "✓" if correct else "✗"
            print(f"  {status} {audio_file.name}: {confidence:.4f} ({'detected' if detected else 'not detected'})")

        except Exception as e:
            print(f"  ✗ {audio_file.name}: Error - {e}")
            results.append({
                "file": audio_file.name,
                "confidence": 0,
                "detected": False,
                "expected": expected_result,
                "correct": False,
                "error": str(e),
            })

    correct_count = sum(1 for r in results if r["correct"])

    return {
        "total": len(results),
        "correct": correct_count,
        "accuracy": correct_count / len(results) if results else 0,
        "files": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch test wake word detection")
    parser.add_argument(
        "--model",
        type=str,
        default="models/wake_word_model.onnx",
        help="Path to wake word model",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="tests",
        help="Test directory containing positive/ and negative/ subdirs",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection threshold",
    )
    args = parser.parse_args()

    test_path = Path(args.test_dir)
    positive_dir = test_path / "positive"
    negative_dir = test_path / "negative"

    # 检查目录
    if not test_path.exists():
        print(f"Error: Test directory '{test_path}' not found")
        print(f"\nPlease create the following structure:")
        print(f"  {test_path}/")
        print(f"  ├── positive/   # Put wake word audio here")
        print(f"  └── negative/   # Put non-wake word audio here")
        sys.exit(1)

    # 加载模型
    print(f"Loading model: {args.model}")
    print(f"Threshold: {args.threshold}")
    print("=" * 60)

    try:
        detector = WakeWordDetector(
            model_path=args.model,
            threshold=args.threshold,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nMake sure you have trained the model first:")
        print("  python -m src.training.train")
        sys.exit(1)

    # 测试正样本
    print("\n[Positive Samples] (should be detected)")
    print("-" * 60)
    if positive_dir.exists():
        positive_results = test_directory(
            detector, positive_dir, expected_result=True, threshold=args.threshold
        )
    else:
        print(f"  Directory not found: {positive_dir}")
        positive_results = {"total": 0, "correct": 0, "accuracy": 0}

    # 测试负样本
    print("\n[Negative Samples] (should NOT be detected)")
    print("-" * 60)
    if negative_dir.exists():
        negative_results = test_directory(
            detector, negative_dir, expected_result=False, threshold=args.threshold
        )
    else:
        print(f"  Directory not found: {negative_dir}")
        negative_results = {"total": 0, "correct": 0, "accuracy": 0}

    # 统计结果
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_positive = positive_results["total"]
    correct_positive = positive_results["correct"]
    total_negative = negative_results["total"]
    correct_negative = negative_results["correct"]

    print(f"\nPositive samples (wake words):")
    print(f"  Total: {total_positive}")
    print(f"  Correct: {correct_positive}")
    if total_positive > 0:
        recall = correct_positive / total_positive
        print(f"  Recall (sensitivity): {recall:.1%}")

    print(f"\nNegative samples (non-wake words):")
    print(f"  Total: {total_negative}")
    print(f"  Correct: {correct_negative}")
    if total_negative > 0:
        specificity = correct_negative / total_negative
        print(f"  Specificity: {specificity:.1%}")

    # 总体准确率
    total = total_positive + total_negative
    correct = correct_positive + correct_negative
    if total > 0:
        accuracy = correct / total
        print(f"\nOverall:")
        print(f"  Total: {total}")
        print(f"  Correct: {correct}")
        print(f"  Accuracy: {accuracy:.1%}")

        # 计算 F1
        if total_positive > 0 and (correct_positive + (total_negative - correct_negative)) > 0:
            precision = correct_positive / (correct_positive + (total_negative - correct_negative))
            recall = correct_positive / total_positive
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            print(f"  Precision: {precision:.1%}")
            print(f"  F1 Score: {f1:.1%}")

    print("\n" + "=" * 60)

    # 建议
    if total_positive > 0 and correct_positive / total_positive < 0.8:
        print("\n⚠️  Recall is low. Consider:")
        print("   - Lowering the threshold")
        print("   - Adding more training data")

    if total_negative > 0 and correct_negative / total_negative < 0.8:
        print("\n⚠️  Too many false positives. Consider:")
        print("   - Raising the threshold")
        print("   - Adding more negative training samples")


if __name__ == "__main__":
    main()
