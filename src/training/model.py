"""
Wake Word Detection Model

轻量级唤醒词检测模型，适合在 CPU 上实时运行
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积 - 减少参数量"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x)


class WakeWordModel(nn.Module):
    """
    轻量级唤醒词检测模型

    架构: CNN + GRU + FC
    - 使用深度可分离卷积减少参数
    - GRU 捕捉时序信息
    - 全连接层输出二分类结果
    """

    def __init__(
        self,
        n_mels: int = 40,
        hidden_size: int = 64,
        num_classes: int = 1,
    ):
        super().__init__()

        self.n_mels = n_mels

        # CNN 特征提取
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.conv2 = DepthwiseSeparableConv(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = DepthwiseSeparableConv(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # 计算 CNN 输出维度
        # 输入: (batch, 1, n_mels, time)
        # conv1 + pool: (batch, 16, n_mels/2, time/2)
        # conv2 + pool: (batch, 32, n_mels/4, time/4)
        # conv3 + pool: (batch, 64, n_mels/8, time/8)
        cnn_out_freq = n_mels // 8
        cnn_out_channels = 64

        # GRU 时序建模
        self.gru = nn.GRU(
            input_size=cnn_out_channels * cnn_out_freq,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # 全连接分类器
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Mel 频谱图 (batch, n_mels, time)

        Returns:
            logits: (batch, 1)
        """
        # 添加通道维度
        x = x.unsqueeze(1)  # (batch, 1, n_mels, time)

        # CNN 特征提取
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)

        # 重排为时序格式: (batch, time, features)
        batch, channels, freq, time = x.shape
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, freq)
        x = x.reshape(batch, time, channels * freq)

        # GRU
        x, _ = self.gru(x)

        # 取最后一个时间步的输出
        x = x[:, -1, :]

        # 分类
        x = self.fc(x)

        return x


class WakeWordModelSmall(nn.Module):
    """
    超轻量级唤醒词检测模型

    更小的模型，适合资源受限的环境
    """

    def __init__(self, n_mels: int = 40, num_classes: int = 1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (batch, 1, n_mels, time)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    n_mels: int = 40,
    time_steps: int = 100,
):
    """
    导出模型为 ONNX 格式

    Args:
        model: PyTorch 模型
        output_path: 输出路径
        n_mels: Mel 频带数
        time_steps: 时间步数
    """
    # 移到 CPU 导出（ONNX 推理通常在 CPU）
    model = model.cpu()
    model.eval()

    # 创建示例输入
    dummy_input = torch.randn(1, n_mels, time_steps)

    # 导出
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "time"},
            "output": {0: "batch_size"},
        },
    )

    print(f"Model exported to {output_path}")


if __name__ == "__main__":
    # 测试模型
    model = WakeWordModel(n_mels=40, hidden_size=64)
    print(f"WakeWordModel parameters: {count_parameters(model):,}")

    model_small = WakeWordModelSmall(n_mels=40)
    print(f"WakeWordModelSmall parameters: {count_parameters(model_small):,}")

    # 测试前向传播
    x = torch.randn(2, 40, 100)  # (batch, n_mels, time)
    y = model(x)
    print(f"Output shape: {y.shape}")

    y_small = model_small(x)
    print(f"Output shape (small): {y_small.shape}")
