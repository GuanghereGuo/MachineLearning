import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class TwoLayerNetPyTorch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_ratio=0.5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    epoch_num = 20
    batch_size = 100
    hidden_size = 256
    learning_rate = 0.001
    dropout_ratio = 0.5
    input_size = 28 * 28  # 784
    output_size = 10

    # --- 数据加载和预处理 ---
    # 使用 torchvision 加载 MNIST 数据集
    # ToTensor() 会将 PIL Image 或 numpy.ndarray 转换为 FloatTensor，并将像素值从 [0, 255] 缩放到 [0.0, 1.0]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # 可以选择性地添加归一化，这通常能提升性能
            # transforms.Normalize((0.1307,), (0.3081,)) # MNIST 数据集的均值和标准差
        ]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # 使用 DataLoader 进行批量加载和打乱
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = TwoLayerNetPyTorch(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        dropout_ratio=dropout_ratio,
    ).to(device)

    # nn.CrossEntropyLoss 内部包含了 Softmax，因此模型输出层不需要 Softmax 激活函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for epoch in range(epoch_num):
        model.train()  # 设置为训练模式 (启用 Dropout 和 BN 的训练行为)
        running_loss = 0.0

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epoch_num} [Training]"
        )
        for inputs, labels in progress_bar:
            # 将数据移动到 GPU
            inputs = inputs.view(inputs.shape[0], -1).to(
                device
            )  # 展平图像: [B, 1, 28, 28] -> [B, 784]
            labels = labels.to(device)

            # 1. 梯度清零
            optimizer.zero_grad()

            # 2. 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 3. 反向传播
            loss.backward()

            # 4. 更新权重
            optimizer.step()

            running_loss += loss.item()
            train_loss_list.append(loss.item())

        # --- 评估阶段 ---
        model.eval()  # 设置为评估模式 (禁用 Dropout 和 BN 的评估行为)
        train_correct = 0
        train_total = 0
        test_correct = 0
        test_total = 0

        # 使用 torch.no_grad() 上下文管理器，在该代码块内禁用梯度计算，以节省内存和计算资源
        with torch.no_grad():
            # 计算训练集准确率
            for inputs, labels in train_loader:
                inputs = inputs.view(inputs.shape[0], -1).to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            # 计算测试集准确率
            for inputs, labels in test_loader:
                inputs = inputs.view(inputs.shape[0], -1).to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total
        test_acc = 100 * test_correct / test_total

        train_acc_list.append(train_acc / 100.0)  # 转换为 0-1 范围以匹配绘图
        test_acc_list.append(test_acc / 100.0)

        avg_epoch_loss = running_loss / len(train_loader)
        print(
            f"Epoch {epoch + 1}/{epoch_num} done. "
            f"Avg Loss: {avg_epoch_loss:.4f}, "
            f"Train Acc: {train_acc:.2f}%, "
            f"Test Acc: {test_acc:.2f}%"
        )

    # --- 结果可视化 ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # 绘制每个 iteration 的 loss
    plt.plot(train_loss_list)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    plt.subplot(1, 2, 2)
    # 绘制每个 epoch 的 accuracy
    plt.plot(train_acc_list, label="Train Accuracy")
    plt.plot(test_acc_list, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Training complete.")
    print(
        f"Final Train Accuracy: {train_acc_list[-1]:.4f}, "
        f"Final Test Accuracy: {test_acc_list[-1]:.4f}"
    )
