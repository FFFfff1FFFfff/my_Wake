"""
Wake Word Detector

实时唤醒词检测模块，支持 ONNX 和 PyTorch 模型
"""

import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


class AudioFeatureExtractor:
    """音频特征提取器 - 与训练时使用相同的配置"""

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
        self.mel_basis = self._create_mel_filterbank()

    def _create_mel_filterbank(self) -> np.ndarray:
        """创建 Mel 滤波器组"""

        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        mel_min = hz_to_mel(self.fmin)
        mel_max = hz_to_mel(self.fmax)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        bin_points = np.floor(
            (self.n_fft + 1) * hz_points / self.sample_rate
        ).astype(int)

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
        """提取 Mel 频谱图特征"""
        # 预加重
        pre_emphasis = 0.97
        emphasized = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

        # 分帧
        num_frames = 1 + (len(emphasized) - self.win_length) // self.hop_length
        if num_frames <= 0:
            return np.zeros((self.n_mels, 1))

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


class WakeWordDetector:
    """
    唤醒词检测器

    支持实时音频流检测，当检测到唤醒词时触发回调
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        window_duration: float = 2.0,
        hop_duration: float = 0.1,
        smoothing_window: int = 3,
    ):
        """
        Args:
            model_path: ONNX 模型路径
            threshold: 检测阈值 (0-1)
            sample_rate: 采样率
            window_duration: 检测窗口长度（秒）
            hop_duration: 检测步长（秒）
            smoothing_window: 平滑窗口大小
        """
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.window_samples = int(window_duration * sample_rate)
        self.hop_samples = int(hop_duration * sample_rate)
        self.smoothing_window = smoothing_window

        # 加载模型
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime is required for inference")

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

        # 特征提取器
        self.feature_extractor = AudioFeatureExtractor(sample_rate=sample_rate)

        # 音频缓冲区
        self.audio_buffer = np.zeros(self.window_samples, dtype=np.float32)

        # 检测历史（用于平滑）
        self.detection_history = []

        # 回调函数
        self.on_wake_word: Optional[Callable[[], None]] = None

        # 运行状态
        self._running = False
        self._audio_queue = queue.Queue()

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid 函数"""
        return 1 / (1 + np.exp(-x))

    def detect(self, audio: np.ndarray) -> float:
        """
        检测音频中是否包含唤醒词

        Args:
            audio: 音频波形

        Returns:
            检测置信度 (0-1)
        """
        # 提取特征
        features = self.feature_extractor.extract(audio)

        # 添加 batch 维度
        features = features[np.newaxis, :, :].astype(np.float32)

        # 推理
        outputs = self.session.run(None, {self.input_name: features})
        logits = outputs[0][0, 0]

        # Sigmoid 转换为概率
        confidence = float(self._sigmoid(logits))

        return confidence

    def process_audio(self, audio_chunk: np.ndarray) -> Optional[float]:
        """
        处理新的音频块

        Args:
            audio_chunk: 新的音频数据

        Returns:
            如果检测到唤醒词，返回置信度；否则返回 None
        """
        # 更新缓冲区
        chunk_len = len(audio_chunk)
        if chunk_len >= self.window_samples:
            self.audio_buffer = audio_chunk[-self.window_samples :]
        else:
            self.audio_buffer = np.concatenate(
                [self.audio_buffer[chunk_len:], audio_chunk]
            )

        # 检测
        confidence = self.detect(self.audio_buffer)

        # 更新历史
        self.detection_history.append(confidence)
        if len(self.detection_history) > self.smoothing_window:
            self.detection_history.pop(0)

        # 平滑后的置信度
        smoothed_confidence = np.mean(self.detection_history)

        # 检查是否触发
        if smoothed_confidence > self.threshold:
            # 清空历史，防止重复触发
            self.detection_history = []
            return smoothed_confidence

        return None

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice 音频回调"""
        if status:
            print(f"Audio status: {status}")
        self._audio_queue.put(indata.copy().flatten())

    def _detection_loop(self):
        """检测循环"""
        while self._running:
            try:
                audio_chunk = self._audio_queue.get(timeout=0.1)
                result = self.process_audio(audio_chunk)

                if result is not None:
                    print(f"\n[Wake Word Detected! Confidence: {result:.2f}]")
                    if self.on_wake_word:
                        self.on_wake_word()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Detection error: {e}")

    def start_listening(self, on_wake_word: Optional[Callable[[], None]] = None):
        """
        开始监听麦克风

        Args:
            on_wake_word: 检测到唤醒词时的回调函数
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError("sounddevice is required for microphone input")

        self.on_wake_word = on_wake_word
        self._running = True

        # 启动检测线程
        detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        detection_thread.start()

        # 启动音频流
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=self.hop_samples,
            callback=self._audio_callback,
        ):
            print("Listening for wake word... (Press Ctrl+C to stop)")
            try:
                while self._running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nStopping...")

        self._running = False

    def stop_listening(self):
        """停止监听"""
        self._running = False


class WakeWordDetectorWithVAD(WakeWordDetector):
    """
    带 VAD 的唤醒词检测器

    检测到唤醒词后自动录制后续语音
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        window_duration: float = 2.0,
        hop_duration: float = 0.1,
        max_recording_duration: float = 10.0,
        silence_threshold: float = 0.01,
        silence_duration: float = 1.0,
    ):
        super().__init__(
            model_path=model_path,
            threshold=threshold,
            sample_rate=sample_rate,
            window_duration=window_duration,
            hop_duration=hop_duration,
        )

        self.max_recording_samples = int(max_recording_duration * sample_rate)
        self.silence_threshold = silence_threshold
        self.silence_samples = int(silence_duration * sample_rate)

        self.is_recording = False
        self.recording_buffer = []
        self.silence_counter = 0

        self.on_recording_complete: Optional[Callable[[np.ndarray], None]] = None

    def _is_silence(self, audio: np.ndarray) -> bool:
        """检测是否为静音"""
        return np.abs(audio).mean() < self.silence_threshold

    def _detection_loop(self):
        """检测循环（包含录音逻辑）"""
        while self._running:
            try:
                audio_chunk = self._audio_queue.get(timeout=0.1)

                if self.is_recording:
                    # 录音中
                    self.recording_buffer.append(audio_chunk)

                    # 检测静音
                    if self._is_silence(audio_chunk):
                        self.silence_counter += len(audio_chunk)
                    else:
                        self.silence_counter = 0

                    # 检查是否结束录音
                    total_samples = sum(len(chunk) for chunk in self.recording_buffer)
                    if (
                        self.silence_counter >= self.silence_samples
                        or total_samples >= self.max_recording_samples
                    ):
                        # 录音结束
                        self.is_recording = False
                        recording = np.concatenate(self.recording_buffer)
                        self.recording_buffer = []
                        self.silence_counter = 0

                        print(f"[Recording complete: {len(recording) / self.sample_rate:.1f}s]")

                        if self.on_recording_complete:
                            self.on_recording_complete(recording)

                else:
                    # 检测唤醒词
                    result = self.process_audio(audio_chunk)

                    if result is not None:
                        print(f"\n[Wake Word Detected! Confidence: {result:.2f}]")
                        print("[Recording...]")

                        # 开始录音
                        self.is_recording = True
                        self.recording_buffer = []
                        self.silence_counter = 0

                        if self.on_wake_word:
                            self.on_wake_word()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Detection error: {e}")

    def start_listening(
        self,
        on_wake_word: Optional[Callable[[], None]] = None,
        on_recording_complete: Optional[Callable[[np.ndarray], None]] = None,
    ):
        """
        开始监听麦克风

        Args:
            on_wake_word: 检测到唤醒词时的回调
            on_recording_complete: 录音完成时的回调，参数为音频数据
        """
        self.on_recording_complete = on_recording_complete
        super().start_listening(on_wake_word)


def main():
    """测试唤醒词检测器"""
    import argparse

    parser = argparse.ArgumentParser(description="Wake word detector")
    parser.add_argument(
        "--model",
        type=str,
        default="models/wake_word_model.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection threshold",
    )
    args = parser.parse_args()

    detector = WakeWordDetector(
        model_path=args.model,
        threshold=args.threshold,
    )

    def on_wake():
        print("Wake word callback triggered!")

    detector.start_listening(on_wake_word=on_wake)


if __name__ == "__main__":
    main()
