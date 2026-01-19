"""
Training Script for Wake Word Detection Model

训练唤醒词检测模型的主脚本
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.dataset import WakeWordDataset, create_dataloaders
from src.training.model import WakeWordModel, WakeWordModelSmall, count_parameters, export_to_onnx


class Trainer:
    """模型训练器"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # 损失函数 - 使用带权重的 BCE 处理类别不平衡
        self.criterion = nn.BCEWithLogitsLoss()

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        # 记录最佳模型
        self.best_val_loss = float("inf")
        self.best_model_state = None

    def train_epoch(self) -> Tuple[float, float]:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for features, labels in pbar:
            features = features.to(self.device)
            labels = labels.float().to(self.device).unsqueeze(1)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 统计
            total_loss += loss.item() * features.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += features.size(0)

            pbar.set_postfix({"loss": loss.item(), "acc": correct / total})

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float, Dict]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        # 用于计算详细指标
        all_labels = []
        all_predictions = []
        all_scores = []

        with torch.no_grad():
            for features, labels in tqdm(self.val_loader, desc="Validating", leave=False):
                features = features.to(self.device)
                labels = labels.float().to(self.device).unsqueeze(1)

                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * features.size(0)
                scores = torch.sigmoid(outputs)
                predictions = (scores > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += features.size(0)

                all_labels.extend(labels.cpu().numpy().flatten())
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_scores.extend(scores.cpu().numpy().flatten())

        avg_loss = total_loss / total
        accuracy = correct / total

        # 计算详细指标
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        tp = ((all_predictions == 1) & (all_labels == 1)).sum()
        fp = ((all_predictions == 1) & (all_labels == 0)).sum()
        fn = ((all_predictions == 0) & (all_labels == 1)).sum()
        tn = ((all_predictions == 0) & (all_labels == 0)).sum()

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

        return avg_loss, accuracy, metrics

    def train(
        self,
        num_epochs: int,
        save_dir: str,
        early_stopping_patience: int = 10,
    ) -> Dict:
        """
        完整训练流程

        Args:
            num_epochs: 训练轮数
            save_dir: 模型保存目录
            early_stopping_patience: 早停耐心值

        Returns:
            训练历史记录
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
        }

        patience_counter = 0

        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Device: {self.device}")
        print("-" * 50)

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # 训练
            train_loss, train_acc = self.train_epoch()

            # 验证
            val_loss, val_acc, metrics = self.validate()

            # 更新学习率
            self.scheduler.step(val_loss)

            # 记录历史
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(metrics["f1"])

            # 打印结果
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(
                f"  Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}"
            )

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0

                # 保存检查点
                checkpoint_path = save_path / "best_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "metrics": metrics,
                    },
                    checkpoint_path,
                )
                print(f"  Saved best model to {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        # 导出 ONNX
        onnx_path = save_path / "wake_word_model.onnx"
        export_to_onnx(self.model, str(onnx_path))

        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        return history


def main():
    parser = argparse.ArgumentParser(description="Train wake word detection model")
    parser.add_argument(
        "--positive-dirs",
        type=str,
        nargs="+",
        default=["data/positive/hi_iroi", "data/positive/hey_iroi", "data/positive/hello_iroi"],
        help="Directories containing positive samples",
    )
    parser.add_argument(
        "--negative-dir",
        type=str,
        default="data/negative",
        help="Directory containing negative samples",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["standard", "small"],
        default="standard",
        help="Model type (standard or small)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    args = parser.parse_args()

    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(
        positive_dirs=args.positive_dirs,
        negative_dir=args.negative_dir,
        batch_size=args.batch_size,
    )

    # 创建模型
    if args.model_type == "small":
        model = WakeWordModelSmall(n_mels=40)
    else:
        model = WakeWordModel(n_mels=40, hidden_size=64)

    print(f"Model type: {args.model_type}")
    print(f"Parameters: {count_parameters(model):,}")

    # 创建训练器
    device = torch.device(args.device)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
    )

    # 开始训练
    history = trainer.train(
        num_epochs=args.epochs,
        save_dir=args.save_dir,
    )

    print("\nTraining history:")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"  Final val F1: {history['val_f1'][-1]:.4f}")


if __name__ == "__main__":
    main()
