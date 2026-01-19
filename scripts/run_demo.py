#!/usr/bin/env python3
"""
语音助手演示脚本

演示唤醒词检测 + OpenAI ASR 的完整流程
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def run_wake_word_only(model_path: str, threshold: float):
    """仅运行唤醒词检测"""
    from src.inference.wake_detector import WakeWordDetector

    print("=" * 50)
    print("Wake Word Detection Demo")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Threshold: {threshold}")
    print("-" * 50)

    detector = WakeWordDetector(
        model_path=model_path,
        threshold=threshold,
    )

    def on_wake():
        print("\n>>> Wake word detected! <<<\n")

    detector.start_listening(on_wake_word=on_wake)


def run_with_asr(model_path: str, threshold: float, language: str = None):
    """运行唤醒词检测 + ASR"""
    from src.inference.openai_asr import ASRWithWakeWord

    print("=" * 50)
    print("Voice Assistant Demo (Wake Word + ASR)")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Threshold: {threshold}")
    print(f"Language: {language or 'auto'}")
    print("-" * 50)

    assistant = ASRWithWakeWord(
        wake_word_model_path=model_path,
        wake_threshold=threshold,
        language=language,
    )

    def on_transcription(text: str):
        print(f"\n>>> User said: {text} <<<\n")
        # 这里可以添加你的业务逻辑
        # 比如发送到 LLM 获取回复

    assistant.start(on_transcription=on_transcription)


def run_test_audio(model_path: str, audio_path: str, threshold: float):
    """测试单个音频文件"""
    import numpy as np
    import soundfile as sf
    from scipy import signal

    from src.inference.wake_detector import WakeWordDetector

    print("=" * 50)
    print("Testing audio file")
    print("=" * 50)
    print(f"Audio: {audio_path}")
    print(f"Model: {model_path}")
    print("-" * 50)

    # 加载音频
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # 重采样到 16kHz
    if sr != 16000:
        num_samples = int(len(audio) * 16000 / sr)
        audio = signal.resample(audio, num_samples)

    # 检测
    detector = WakeWordDetector(
        model_path=model_path,
        threshold=threshold,
    )

    confidence = detector.detect(audio)
    is_wake_word = confidence > threshold

    print(f"Confidence: {confidence:.4f}")
    print(f"Is wake word: {is_wake_word}")


def main():
    parser = argparse.ArgumentParser(description="Voice assistant demo")
    parser.add_argument(
        "--model",
        type=str,
        default="models/wake_word_model.onnx",
        help="Path to wake word model",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection threshold",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["wake-only", "with-asr", "test"],
        default="with-asr",
        help="Demo mode",
    )
    parser.add_argument(
        "--audio",
        type=str,
        help="Audio file to test (for test mode)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="ASR language code (e.g., 'en', 'zh')",
    )
    args = parser.parse_args()

    if args.mode == "wake-only":
        run_wake_word_only(args.model, args.threshold)
    elif args.mode == "with-asr":
        run_with_asr(args.model, args.threshold, args.language)
    elif args.mode == "test":
        if not args.audio:
            print("Error: --audio is required for test mode")
            sys.exit(1)
        run_test_audio(args.model, args.audio, args.threshold)


if __name__ == "__main__":
    main()
