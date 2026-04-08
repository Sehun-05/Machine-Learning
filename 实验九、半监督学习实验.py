import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet34
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import warnings

warnings.filterwarnings("ignore")

# -------------------------- 1. 配置参数 --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LABELED_RATIO = 0.1
EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PSEUDO_THRESHOLD_INIT = 0.7
PSEUDO_THRESHOLD_MAX = 0.9
TEMPERATURE = 0.5
PSEUDO_UPDATE_INTERVAL = 10
PATIENCE = 15
MIXED_PRECISION = True if DEVICE.type == "cuda" else False # CPU不支持混合精度

# 路径配置
DATA_DIR = "D:\\pycharm项目\\PythonProject1\\data"
LOG_DIR = "D:\\tb_logs_optimized"
os.makedirs(LOG_DIR, exist_ok=True)
print(f"设备: {DEVICE} | 日志目录: {LOG_DIR}")

# 训练日志存储
train_logs = {
    "total_loss": [], "labeled_loss": [], "unlabeled_loss": [],
    "learning_rate": [], "test_accuracy": [], "pseudo_threshold": []
}

# -------------------------- 2. 修正的数据预处理 --------------------------
# 弱增强，用于生成伪标签
transform_weak = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 强增强，用于训练
transform_strong = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),  # 移到这里
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.33)), # 现在它接收的是Tensor
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 现在它接收的是Tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 加载数据集
full_train_dataset_strong = datasets.CIFAR10(root=DATA_DIR, train=True, download=False, transform=transform_strong)
full_train_dataset_weak = datasets.CIFAR10(root=DATA_DIR, train=True, download=False, transform=transform_weak)
test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=False, transform=transform_test)

# 划分有标签/无标签数据
num_total = len(full_train_dataset_strong)
num_labeled = int(num_total * LABELED_RATIO)
labeled_indices = np.random.choice(num_total, num_labeled, replace=False)
unlabeled_indices = np.setdiff1d(np.arange(num_total), labeled_indices)

labeled_dataset = Subset(full_train_dataset_strong, labeled_indices)
unlabeled_dataset = Subset(full_train_dataset_strong, unlabeled_indices)

# 数据加载器
labeled_loader = DataLoader(labeled_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# -------------------------- 3. 模型定义 --------------------------
class CIFAR10Model(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = resnet34(weights=None)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.fc = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

model = CIFAR10Model().to(DEVICE)
print("模型结构: ResNet-34 + BatchNorm + Dropout")

# -------------------------- 4. 损失函数与优化器 --------------------------
criterion_labeled = nn.CrossEntropyLoss()
criterion_unlabeled = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - 5, eta_min=1e-5)

scaler = GradScaler() if MIXED_PRECISION else None

# -------------------------- 5. 迭代更新伪标签 --------------------------
def update_pseudo_labels(model, unlabeled_indices, current_threshold):
    model.eval()
    valid_unlabeled_indices = []
    with torch.no_grad():
        for idx in unlabeled_indices:
            # full_train_dataset_weak 的 transform 已经包含了 ToTensor 和 Normalize
            img, _ = full_train_dataset_weak[idx]
            img = img.unsqueeze(0).to(DEVICE)
            with autocast(enabled=MIXED_PRECISION):
                output = model(img)
            pseudo_probs = torch.softmax(output / TEMPERATURE, dim=1)
            if pseudo_probs.max().item() >= current_threshold:
                valid_unlabeled_indices.append(idx)
    print(f"伪标签更新: 保留 {len(valid_unlabeled_indices)}/{len(unlabeled_indices)} 个高置信度样本")
    return valid_unlabeled_indices

# -------------------------- 6. 日志与可视化 --------------------------
def save_logs():
    log_path = os.path.join(LOG_DIR, "train_logs_optimized.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(train_logs, f, indent=2)

def plot_metrics():
    if not train_logs["total_loss"]: return
    epochs = range(1, len(train_logs["total_loss"]) + 1)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    ax1.plot(epochs, train_logs["total_loss"], label="Total Loss"); ax1.set_title("Total Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.6)
    ax2.plot(epochs, train_logs["labeled_loss"], label="Labeled Loss"); ax2.plot(epochs, train_logs["unlabeled_loss"], label="Unlabeled Loss"); ax2.set_title("Labeled vs Unlabeled Loss"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.6)
    ax3.plot(epochs, train_logs["test_accuracy"], label="Test Accuracy", color='green'); ax3.set_title("Test Accuracy"); ax3.set_xlabel("Epoch"); ax3.set_ylabel("Accuracy (%)"); ax3.legend(); ax3.grid(True, linestyle='--', alpha=0.6)
    ax4.plot(epochs, train_logs["pseudo_threshold"], label="Pseudo Threshold", color='purple'); ax4.set_title("Dynamic Pseudo Threshold"); ax4.set_xlabel("Epoch"); ax4.set_ylabel("Threshold"); ax4.legend(); ax4.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plot_path = os.path.join(LOG_DIR, "metrics_curve_optimized.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

# -------------------------- 7. 训练函数 --------------------------
def train_epoch(epoch, current_pseudo_threshold):
    model.train()
    total_loss, labeled_loss_sum, unlabeled_loss_sum, total_steps = 0.0, 0.0, 0.0, 0

    unlabeled_iter = iter(unlabeled_loader)
    for labeled_imgs, labeled_labels in labeled_loader:
        labeled_imgs, labeled_labels = labeled_imgs.to(DEVICE), labeled_labels.to(DEVICE)

        # --- 有标签数据损失 ---
        with autocast(enabled=MIXED_PRECISION):
            labeled_outputs = model(labeled_imgs)
            labeled_loss = criterion_labeled(labeled_outputs, labeled_labels)

        # --- 无标签数据损失 ---
        try:
            unlabeled_imgs, _ = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            unlabeled_imgs, _ = next(unlabeled_iter)
        unlabeled_imgs = unlabeled_imgs.to(DEVICE)

        with torch.no_grad():
            with autocast(enabled=MIXED_PRECISION):
                unlabeled_outputs = model(unlabeled_imgs)
            pseudo_probs = torch.softmax(unlabeled_outputs / TEMPERATURE, dim=1)
            pseudo_labels = torch.argmax(pseudo_probs, dim=1)
            conf_mask = (pseudo_probs.max(dim=1)[0] >= current_pseudo_threshold).float()

        with autocast(enabled=MIXED_PRECISION):
            unlabeled_outputs_train = model(unlabeled_imgs)
            unlabeled_loss = criterion_unlabeled(unlabeled_outputs_train, pseudo_labels)
            unlabeled_loss = (unlabeled_loss * conf_mask).mean()

        # --- 总损失与优化 ---
        total_loss_batch = labeled_loss + 0.3 * unlabeled_loss

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(total_loss_batch).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss_batch.backward()
            optimizer.step()

        # --- 累计损失 ---
        total_loss += total_loss_batch.item()
        labeled_loss_sum += labeled_loss.item()
        unlabeled_loss_sum += unlabeled_loss.item()
        total_steps += 1

    avg_total_loss = total_loss / total_steps if total_steps > 0 else 0.0
    avg_labeled_loss = labeled_loss_sum / total_steps if total_steps > 0 else 0.0
    avg_unlabeled_loss = unlabeled_loss_sum / total_steps if total_steps > 0 else 0.0

    train_logs["total_loss"].append(avg_total_loss)
    train_logs["labeled_loss"].append(avg_labeled_loss)
    train_logs["unlabeled_loss"].append(avg_unlabeled_loss)
    train_logs["learning_rate"].append(optimizer.param_groups[0]["lr"])
    train_logs["pseudo_threshold"].append(current_pseudo_threshold)

    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
    print(f"Loss: Total={avg_total_loss:.4f} | Labeled={avg_labeled_loss:.4f} | Unlabeled={avg_unlabeled_loss:.4f}")
    print(f"LR: {optimizer.param_groups[0]['lr']:.6f} | Pseudo Threshold: {current_pseudo_threshold:.2f}")

# -------------------------- 8. 评估函数 --------------------------
def evaluate():
    model.eval()
    correct, total = 0, 0
    all_preds, all_true = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with autocast(enabled=MIXED_PRECISION):
                outputs = model(imgs)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_true.extend(labels.cpu().numpy().tolist())

    accuracy = 100 * correct / total
    train_logs["test_accuracy"].append(accuracy)
    print(f"Test Accuracy: {accuracy:.2f}%")

    preds_path = os.path.join(LOG_DIR, "test_predictions_optimized.json")
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump({"predictions": all_preds, "true_labels": all_true, "best_accuracy": accuracy}, f, indent=2)
    return accuracy, all_preds, all_true

# -------------------------- 9. 主训练流程 --------------------------
if __name__ == "__main__":
    best_accuracy, best_preds, best_true = 0.0, [], []
    patience_counter = 0

    print("=" * 60)
    print("开始优化版训练 (Pseudo Label)")
    print(f"有标签数据: {len(labeled_dataset)} | 初始无标签数据: {len(unlabeled_dataset)} | 测试集: {len(test_dataset)}")
    print("=" * 60)

    for epoch in range(EPOCHS):
        current_pseudo_threshold = PSEUDO_THRESHOLD_INIT + (epoch / EPOCHS) * (PSEUDO_THRESHOLD_MAX - PSEUDO_THRESHOLD_INIT)

        train_epoch(epoch, current_pseudo_threshold)
        test_acc, test_preds, test_true = evaluate()

        if epoch < 5: scheduler_warmup.step()
        else: scheduler_cosine.step()

        if test_acc > best_accuracy:
            best_accuracy, best_preds, best_true = test_acc, test_preds, test_true
            patience_counter = 0
            model_path = os.path.join(LOG_DIR, "best_model_optimized.pth")
            torch.save(model.state_dict(), model_path)
            print(f"✅ 保存最优模型 (准确率: {best_accuracy:.2f}%)")
        else:
            patience_counter += 1
            print(f"早停计数器: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("⚠️  早停触发，停止训练。")
                break

        if (epoch + 1) % PSEUDO_UPDATE_INTERVAL == 0:
            print("\n" + "=" * 40)
            print(f"Epoch {epoch + 1}: 更新伪标签...")
            valid_unlabeled_indices = update_pseudo_labels(model, unlabeled_indices, current_pseudo_threshold)
            unlabeled_dataset = Subset(full_train_dataset_strong, valid_unlabeled_indices)
            unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
            print("=" * 40 + "\n")

        if (epoch + 1) % 5 == 0:
            save_logs()
            plot_metrics()
            print(f"📊 日志和图表已保存到: {LOG_DIR}")

    save_logs()
    plot_metrics()

    print("\n" + "=" * 60)
    print("🏆 训练完成!")
    print(f"最佳测试准确率: {best_accuracy:.2f}%")
    print(f"前10个预测: {best_preds[:10]}")
    print(f"前10个真实: {best_true[:10]}")
    print(f"日志文件: {os.path.join(LOG_DIR, 'train_logs_optimized.json')}")
    print(f"图表文件: {os.path.join(LOG_DIR, 'metrics_curve_optimized.png')}")
    print(f"模型文件: {os.path.join(LOG_DIR, 'best_model_optimized.pth')}")
    print("=" * 60)