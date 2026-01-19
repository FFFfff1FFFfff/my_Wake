# 测试音频目录

## 目录结构

```
tests/
├── positive/    # 正样本（唤醒词，应该被检测到）
│   ├── hi_iroi_1.wav
│   ├── hi_iroi_2.wav
│   ├── hey_iroi_1.wav
│   └── hello_iroi_1.wav
├── negative/    # 负样本（非唤醒词，不应该被检测到）
│   ├── hi_there.wav
│   ├── hello.wav
│   ├── random_speech.wav
│   └── silence.wav
└── README.md
```

## 文件命名建议

- 正样本: `{唤醒词}_{编号}.wav`，如 `hi_iroi_1.wav`
- 负样本: `{内容描述}.wav`，如 `hi_there.wav`

## 支持的格式

WAV, MP3, FLAC, OGG, M4A

## 运行测试

```bash
python scripts/batch_test.py --model models/wake_word_model.onnx
```
