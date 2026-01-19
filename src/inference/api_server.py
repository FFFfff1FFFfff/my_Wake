"""
Wake Word API Server

提供 HTTP API 接口，适用于无声卡的云服务器环境
"""

import io
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from scipy import signal

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from .wake_detector import WakeWordDetector, AudioFeatureExtractor
from .openai_asr import OpenAIASRClient


class WakeWordAPIServer:
    """
    唤醒词检测 HTTP API 服务器

    适用于：
    - 云服务器部署（无声卡）
    - 机器人通过网络发送音频
    - Web 应用集成
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        enable_asr: bool = True,
        openai_api_key: Optional[str] = None,
    ):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required. Install with: pip install fastapi uvicorn python-multipart")

        self.sample_rate = sample_rate
        self.threshold = threshold
        self.enable_asr = enable_asr

        # 初始化唤醒词检测器
        self.detector = WakeWordDetector(
            model_path=model_path,
            threshold=threshold,
            sample_rate=sample_rate,
        )

        # 初始化 ASR（可选）
        self.asr_client = None
        if enable_asr:
            try:
                self.asr_client = OpenAIASRClient(api_key=openai_api_key)
            except Exception as e:
                print(f"Warning: ASR not available: {e}")

        # 创建 FastAPI 应用
        self.app = FastAPI(
            title="Wake Word Detection API",
            description="检测唤醒词 (hi/hey/hello iroi) 并可选进行语音转文字",
            version="1.0.0",
        )

        self._setup_routes()

    def _load_audio(self, audio_bytes: bytes) -> np.ndarray:
        """从字节加载音频"""
        buffer = io.BytesIO(audio_bytes)
        audio, sr = sf.read(buffer)

        # 转单声道
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # 重采样
        if sr != self.sample_rate:
            num_samples = int(len(audio) * self.sample_rate / sr)
            audio = signal.resample(audio, num_samples)

        return audio.astype(np.float32)

    def _setup_routes(self):
        """设置 API 路由"""

        @self.app.get("/")
        async def root():
            return {
                "service": "Wake Word Detection API",
                "wake_words": ["hi iroi", "hey iroi", "hello iroi"],
                "endpoints": {
                    "/detect": "POST - 检测唤醒词",
                    "/transcribe": "POST - 语音转文字（需要先检测到唤醒词）",
                    "/detect_and_transcribe": "POST - 检测唤醒词并转文字",
                }
            }

        @self.app.post("/detect")
        async def detect_wake_word(audio: UploadFile = File(...)):
            """
            检测音频中是否包含唤醒词

            - 支持格式: WAV, MP3, FLAC, OGG
            - 返回: 置信度和是否检测到唤醒词
            """
            try:
                audio_bytes = await audio.read()
                audio_data = self._load_audio(audio_bytes)

                confidence = self.detector.detect(audio_data)
                is_wake_word = confidence > self.threshold

                return {
                    "detected": is_wake_word,
                    "confidence": float(confidence),
                    "threshold": self.threshold,
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.post("/transcribe")
        async def transcribe_audio(audio: UploadFile = File(...)):
            """
            语音转文字（不检测唤醒词）
            """
            if not self.asr_client:
                raise HTTPException(status_code=503, detail="ASR not available")

            try:
                audio_bytes = await audio.read()
                audio_data = self._load_audio(audio_bytes)

                text = self.asr_client.transcribe(audio_data, self.sample_rate)

                return {
                    "text": text,
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.post("/detect_and_transcribe")
        async def detect_and_transcribe(
            audio: UploadFile = File(...),
            transcribe_on_wake: bool = True,
        ):
            """
            检测唤醒词，如果检测到则进行语音转文字

            - transcribe_on_wake: 是否在检测到唤醒词后进行转文字
            """
            try:
                audio_bytes = await audio.read()
                audio_data = self._load_audio(audio_bytes)

                # 检测唤醒词
                confidence = self.detector.detect(audio_data)
                is_wake_word = confidence > self.threshold

                result = {
                    "detected": is_wake_word,
                    "confidence": float(confidence),
                    "threshold": self.threshold,
                    "text": None,
                }

                # 如果检测到唤醒词且需要转文字
                if is_wake_word and transcribe_on_wake and self.asr_client:
                    text = self.asr_client.transcribe(audio_data, self.sample_rate)
                    result["text"] = text

                return result
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """启动服务器"""
        print(f"Starting Wake Word API Server at http://{host}:{port}")
        print(f"Model threshold: {self.threshold}")
        print(f"ASR enabled: {self.asr_client is not None}")
        uvicorn.run(self.app, host=host, port=port)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Wake Word API Server")
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
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port",
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        help="Disable ASR",
    )
    args = parser.parse_args()

    server = WakeWordAPIServer(
        model_path=args.model,
        threshold=args.threshold,
        enable_asr=not args.no_asr,
    )

    server.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
