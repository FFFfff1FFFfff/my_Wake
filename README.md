# my_Wake

自定义唤醒词检测系统，用于检测 "hi iroi"、"hey iroi"、"hello iroi" 等唤醒词，并与 OpenAI Whisper ASR 集成。

## 功能特性

- 支持自定义唤醒词训练（"hi iroi"、"hey iroi"、"hello iroi"）
- 使用 TTS 合成训练数据，无需大量人工录音
- 轻量级模型，可在 CPU 上实时运行
- 与 OpenAI Whisper API 无缝集成
- 支持 VAD（语音活动检测）自动录音

## 项目结构

```
my_Wake/
├── data/
│   ├── positive/           # 正样本（唤醒词音频）
│   │   ├── hi_iroi/
│   │   ├── hey_iroi/
│   │   └── hello_iroi/
│   └── negative/           # 负样本
├── models/
│   └── wake_word_model.onnx
├── src/
│   ├── data_generation/
│   │   ├── tts_generator.py    # TTS 数据生成
│   │   └── audio_augment.py    # 音频增强
│   ├── training/
│   │   ├── dataset.py          # 数据集定义
│   │   ├── model.py            # 模型定义
│   │   └── train.py            # 训练脚本
│   └── inference/
│       ├── wake_detector.py    # 唤醒词检测
│       └── openai_asr.py       # OpenAI ASR 客户端
├── scripts/
│   ├── generate_data.py        # 数据生成脚本
│   └── run_demo.py             # 演示脚本
├── config.yaml
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 设置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，设置 OPENAI_API_KEY
```

### 3. 生成训练数据

```bash
python scripts/generate_data.py
```

这会使用 TTS 生成唤醒词音频样本，并进行数据增强。

### 4. 训练模型

```bash
python -m src.training.train \
    --positive-dirs data/positive/hi_iroi data/positive/hey_iroi data/positive/hello_iroi \
    --negative-dir data/negative \
    --save-dir models \
    --epochs 50
```

### 5. 运行演示

```bash
# 仅唤醒词检测
python scripts/run_demo.py --mode wake-only

# 唤醒词检测 + ASR
python scripts/run_demo.py --mode with-asr

# 测试单个音频文件
python scripts/run_demo.py --mode test --audio path/to/audio.wav
```

## 使用方法

### Python API

```python
from src.inference.openai_asr import ASRWithWakeWord

# 初始化
assistant = ASRWithWakeWord(
    wake_word_model_path="models/wake_word_model.onnx",
    wake_threshold=0.5,
)

# 定义回调
def on_transcription(text):
    print(f"User said: {text}")
    # 处理用户输入...

# 启动
assistant.start(on_transcription=on_transcription)
```

### 仅使用唤醒词检测

```python
from src.inference.wake_detector import WakeWordDetector

detector = WakeWordDetector(
    model_path="models/wake_word_model.onnx",
    threshold=0.5,
)

def on_wake():
    print("Wake word detected!")

detector.start_listening(on_wake_word=on_wake)
```

## 工作流程

```
┌─────────────────────────────────────────────────────────┐
│                      麦克风输入                          │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│          唤醒词检测模型 (持续监听)                        │
│          检测: "hi iroi" / "hey iroi" / "hello iroi"   │
└─────────────────────┬───────────────────────────────────┘
                      │ 检测到唤醒词
                      ▼
┌─────────────────────────────────────────────────────────┐
│          开始录音 + VAD 检测语音结束                      │
└─────────────────────┬───────────────────────────────────┘
                      │ 语音结束
                      ▼
┌─────────────────────────────────────────────────────────┐
│          发送到 OpenAI Whisper API                       │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│          返回转录文本，触发回调                           │
└─────────────────────────────────────────────────────────┘
```

## 模型说明

提供两种模型架构：

### Standard Model (~50K 参数)
- CNN + GRU + FC 架构
- 适合一般场景
- 推理延迟 ~10ms (CPU)

### Small Model (~10K 参数)
- 纯 CNN 架构
- 适合资源受限环境
- 推理延迟 ~5ms (CPU)

## 自定义训练

### 添加新的唤醒词

1. 修改 `src/data_generation/tts_generator.py` 中的 `WAKE_WORDS` 字典：

```python
WAKE_WORDS = {
    "hi_iroi": ["hi iroi", "hi, iroi"],
    "hey_iroi": ["hey iroi", "hey, iroi"],
    "hello_iroi": ["hello iroi", "hello, iroi"],
    "your_new_word": ["your new word", "your, new word"],  # 新增
}
```

2. 重新生成数据并训练

### 调整检测阈值

- 阈值越高，误触发越少，但可能漏检
- 阈值越低，召回率越高，但误触发增加
- 建议从 0.5 开始，根据实际效果调整

## 常见问题

### Q: 没有麦克风怎么测试？

使用测试模式检测音频文件：
```bash
python scripts/run_demo.py --mode test --audio test.wav
```

### Q: 如何提高识别准确率？

1. 收集真实环境的录音作为负样本
2. 增加数据增强的多样性
3. 适当调整检测阈值

### Q: 模型太大怎么办？

使用 small 模型：
```bash
python -m src.training.train --model-type small
```

## License

MIT
