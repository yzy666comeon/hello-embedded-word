import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import utils
from model import CNN

# 配置
LEARNING_RATE = 0.001    # 学习率
BATCH_SIZE = 64          # 批处理量
EPOCHS = 20              # 训练轮数
SEED = 42                # 随机种子
LOG_INTERVAL = 100       # 每训练多少个 batch 打印/记录一次

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets"              # CIFAR-10 数据目录
OUTPUT_DIR = ROOT / "outputs"                # 图片和权重的输出目录
CKPT_PATH = OUTPUT_DIR / "model_para.pth"    # 权重文件

# CIFAR-10 的 10 个类别
CLASSES = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]


# 随机种子
def set_seed(seed):
    """固定随机种子，保证每次运行结果一致。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """有 GPU 就用 GPU，否则用 CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 数据
def get_loaders(batch_size=BATCH_SIZE, download=False):
    """构建训练集和测试集的 DataLoader。"""
    tf = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(
        str(DATASET_DIR), train=True, transform=tf, download=download)
    test_set = torchvision.datasets.CIFAR10(
        str(DATASET_DIR), train=False, transform=tf, download=download)

    # 训练集打乱顺序，测试集不打乱，保证评估顺序固定、结果可复现。
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# 训练
def train_epoch(model, loader, criterion, optimizer, device, epoch, steps, losses, accs):
    """训练一轮，把损失和准确率记录到 steps/losses/accs 三个列表里。"""
    model.train()
    seen = 0          # 本轮已处理的图片数
    run_loss = 0.0    # 当前记录区间累计的损失
    run_correct = 0   # 当前记录区间预测正确的张数
    run_count = 0

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)              # 前向传播
        loss = criterion(logits, labels)    # 计算损失
        optimizer.zero_grad()               # 清空上一轮梯度
        loss.backward()                     # 反向传播
        optimizer.step()                    # 更新参数

        correct = (logits.argmax(1) == labels).sum().item()
        run_loss += loss.item() * labels.size(0)
        run_correct += correct
        run_count += labels.size(0)
        seen += labels.size(0)

        if (i + 1) % LOG_INTERVAL == 0:   #每训练100次，打印一次
            # 横坐标用累计张数
            steps.append((epoch - 1) * len(loader.dataset) + seen)
            losses.append(run_loss / run_count)
            accs.append(run_correct / run_count)
            print(f"  轮次 {epoch} batch {i + 1}/{len(loader)}  "
                  f"损失 {losses[-1]:.4f}  准确率 {accs[-1] * 100:.2f}%")
            run_loss, run_correct, run_count = 0.0, 0, 0

    return seen


# 评估
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """在给定数据集上评估，返回真实标签、预测标签、预测概率、平均损失、准确率。"""
    model.eval()
    trues, preds, scores = [], [], []
    total_loss, total = 0.0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        total += labels.size(0)

        probs = torch.softmax(logits, dim=1)
        trues.append(labels.cpu().numpy())
        preds.append(probs.argmax(1).cpu().numpy())
        scores.append(probs.cpu().numpy())

    y_true = np.concatenate(trues)
    y_pred = np.concatenate(preds)
    return y_true, y_pred, np.concatenate(scores), total_loss / total, utils.accuracy(y_true, y_pred)


# 主流程
def run(args):
    set_seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)
    device = get_device()
    train_loader, test_loader = get_loaders(args.batch_size, args.download)
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # ---------------- 训练 ----------------
    if args.mode in ("train", "all"):
        print(f"开始训练 | 设备 {device} | 学习率 {args.lr} | "
              f"批大小 {args.batch_size} | 轮数 {args.epochs}")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        steps, losses, accs = [], [], []

        for epoch in range(1, args.epochs + 1):
            train_epoch(model, train_loader, criterion, optimizer, device,
                        epoch, steps, losses, accs)
            print(f"第 {epoch}/{args.epochs} 轮结束")
        # 损失/准确率曲线
        utils.plot_curves(steps, losses, accs, OUTPUT_DIR / "train_curves.png", args.show)
        # 保存权重
        torch.save(model.state_dict(), CKPT_PATH)
        print(f"模型已保存: {CKPT_PATH}")

    # ---------------- 评估 ----------------
    if args.mode in ("eval", "all"):
        if args.mode == "eval":
            # map_location 让模型在没有 GPU 的机器上也能加载 GPU 训练的权重
            model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
            print(f"已加载权重: {CKPT_PATH}")

        print("开始在测试集上评估...")
        y_true, y_pred, y_score, loss, acc = evaluate(
            model, test_loader, criterion, device)

        # 混淆矩阵 + ROC 曲线
        cm = utils.confusion_matrix(y_true, y_pred)
        utils.plot_confusion_matrix(cm, CLASSES, OUTPUT_DIR / "confusion_matrix.png", args.show)
        fprs, tprs, aucs, macro = utils.multiclass_roc(y_true, y_score)
        utils.plot_roc(fprs, tprs, aucs, macro, CLASSES, OUTPUT_DIR / "roc.png", args.show)

        # 预测结果预览
        images, labels = next(iter(test_loader))
        utils.plot_predictions(images[:32], labels[:32], y_pred[:32], CLASSES,
                               OUTPUT_DIR / "predictions.png", args.show)

        # 文本报告
        lines = [
            "=" * 62,
            "CIFAR-10 测试集评估报告",
            "=" * 62,
            f"测试样本数       : {len(y_true)}",
            f"Top-1 准确率     : {acc * 100:.2f}%",
            f"平均交叉熵损失   : {loss:.4f}",
            f"宏平均 Macro-AUC : {macro:.4f}",
            "-" * 62,
            f"{'类别':<10}{'召回率':>10}{'AUC':>10}{'样本数':>10}",
            "-" * 62,
        ]
        for c, name in enumerate(CLASSES):
            recall = cm[c, c] / cm[c].sum() * 100 if cm[c].sum() else 0
            auc_text = f"{aucs[c]:.4f}" if aucs[c] is not None else "N/A"
            lines.append(f"{name:<10}{recall:>9.2f}%{auc_text:>10}{cm[c].sum():>10}")
        lines += [
            "-" * 62,
            "召回率 = 该类真实样本中被正确识别的比例",
            "AUC    = 该类别一对其余 ROC 曲线下面积，1 最好，0.5 相当于随机猜",
        ]
        report = "\n".join(lines)
        print(report)
        (OUTPUT_DIR / "report.txt").write_text(report, encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="CIFAR-10 卷积神经网络分类")
    p.add_argument("--mode", choices=["train", "eval", "all"], default="all",
                   help="train 只训练 / eval 只评估 / all 训练后评估")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="训练轮数")
    p.add_argument("--lr", type=float, default=LEARNING_RATE, help="学习率")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="批处理量")
    p.add_argument("--download", action="store_true", help="数据缺失时联网下载")
    p.add_argument("--no-show", dest="show", action="store_false", help="只存图片不弹窗")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())