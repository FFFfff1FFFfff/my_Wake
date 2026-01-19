"""
OpenAI ASR Client

与 OpenAI Whisper API 集成，用于语音转文本
"""

import io
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OpenAIASRClient:
    """
    OpenAI Whisper ASR 客户端

    将音频转换为文本
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "whisper-1",
        language: Optional[str] = None,
    ):
        """
        Args:
            api_key: OpenAI API key，如果为 None 则从环境变量 OPENAI_API_KEY 获取
            model: Whisper 模型名称
            language: 语言代码（如 "en", "zh"），None 表示自动检测
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is required. Install with: pip install openai")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.language = language

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        prompt: Optional[str] = None,
    ) -> str:
        """
        转录音频为文本

        Args:
            audio: 音频波形 (numpy array)
            sample_rate: 采样率
            prompt: 可选的提示词，帮助模型理解上下文

        Returns:
            转录的文本
        """
        # 将 numpy array 转换为音频文件
        audio_bytes = self._audio_to_bytes(audio, sample_rate)

        # 调用 OpenAI API
        response = self.client.audio.transcriptions.create(
            model=self.model,
            file=("audio.wav", audio_bytes, "audio/wav"),
            language=self.language,
            prompt=prompt,
        )

        return response.text

    def transcribe_file(
        self,
        file_path: str,
        prompt: Optional[str] = None,
    ) -> str:
        """
        转录音频文件为文本

        Args:
            file_path: 音频文件路径
            prompt: 可选的提示词

        Returns:
            转录的文本
        """
        with open(file_path, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                language=self.language,
                prompt=prompt,
            )

        return response.text

    def _audio_to_bytes(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """将 numpy array 转换为 WAV 字节"""
        # 确保音频是 float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # 归一化
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.9

        # 写入内存缓冲区
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        buffer.seek(0)

        return buffer.read()


class ASRWithWakeWord:
    """
    结合唤醒词检测和 ASR 的完整语音助手客户端
    """

    def __init__(
        self,
        wake_word_model_path: str,
        openai_api_key: Optional[str] = None,
        wake_threshold: float = 0.5,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ):
        """
        Args:
            wake_word_model_path: 唤醒词模型路径
            openai_api_key: OpenAI API key
            wake_threshold: 唤醒词检测阈值
            sample_rate: 采样率
            language: ASR 语言
        """
        from .wake_detector import WakeWordDetectorWithVAD

        self.sample_rate = sample_rate

        # 初始化唤醒词检测器
        self.wake_detector = WakeWordDetectorWithVAD(
            model_path=wake_word_model_path,
            threshold=wake_threshold,
            sample_rate=sample_rate,
        )

        # 初始化 ASR 客户端
        self.asr_client = OpenAIASRClient(
            api_key=openai_api_key,
            language=language,
        )

        # 回调函数
        self.on_transcription = None
        self.on_wake_word = None

    def _handle_recording(self, audio: np.ndarray):
        """处理录音完成"""
        print("[Transcribing...]")

        try:
            # 调用 ASR
            text = self.asr_client.transcribe(audio, self.sample_rate)
            print(f"[Transcription]: {text}")

            if self.on_transcription:
                self.on_transcription(text)

        except Exception as e:
            print(f"[ASR Error]: {e}")

    def start(self, on_transcription=None, on_wake_word=None):
        """
        启动语音助手

        Args:
            on_transcription: 转录完成回调，参数为转录文本
            on_wake_word: 唤醒词检测回调
        """
        self.on_transcription = on_transcription
        self.on_wake_word = on_wake_word

        print("Voice assistant started!")
        print(f"Say 'hi iroi', 'hey iroi', or 'hello iroi' to activate.")
        print("-" * 50)

        self.wake_detector.start_listening(
            on_wake_word=on_wake_word,
            on_recording_complete=self._handle_recording,
        )

    def stop(self):
        """停止语音助手"""
        self.wake_detector.stop_listening()


def main():
    """测试 ASR 客户端"""
    import argparse

    parser = argparse.ArgumentParser(description="OpenAI ASR client test")
    parser.add_argument(
        "--audio",
        type=str,
        help="Audio file to transcribe",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., 'en', 'zh')",
    )
    args = parser.parse_args()

    client = OpenAIASRClient(language=args.language)

    if args.audio:
        text = client.transcribe_file(args.audio)
        print(f"Transcription: {text}")
    else:
        print("Please provide an audio file with --audio")


if __name__ == "__main__":
    main()
