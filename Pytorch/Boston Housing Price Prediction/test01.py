import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# --- 更改点 1: 导入新的数据集加载函数 ---
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# --- 0. 设置设备 (CUDA 或 CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. 数据加载与预处理 ---
# --- 更改点 2: 使用 fetch_california_housing 加载数据 ---
# 加州房价数据集是一个更现代且无伦理争议的经典回归数据集
housing = fetch_california_housing()
X = housing.data
y = housing.target

y = y.reshape(-1, 1) # 确保目标变量是二维数组

print(f"数据集名称: California Housing")
print(f"原始特征维度: {X.shape}")
print(f"原始目标维度: {y.shape}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放（标准化）
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# 将 NumPy 数组转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# --- CUDA 更改点 1 (不变): 在创建 DataLoader 之前，将整个数据集的张量移动到 GPU ---
# 只有当整个数据集可以适合GPU内存时才这样做！
if device.type == 'cuda': # 仅在GPU可用时进行移动
    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)
    print("整个数据集已移动到GPU。")
else:
    print("数据集保留在CPU。")


# 创建 TensorDataset 和 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# --- 2. 定义神经网络模型 ---
class RegressionNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# 实例化模型
input_dim = X_train_scaled.shape[1]
model = RegressionNet(input_dim)
# --- CUDA 更改点 2 (不变): 将模型移动到指定的设备 (GPU 或 CPU) ---
model.to(device)
print("\n模型结构:")
print(model)

# --- 3. 定义损失函数和优化器 ---
criterion = nn.MSELoss()
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- 4. 训练模型 ---
num_epochs = 200
train_losses = []

print("\n开始训练...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for inputs, targets in train_loader:
        # 数据已预先移动到GPU，这里无需再次移动
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)

    avg_epoch_loss = epoch_loss / len(train_dataset)
    train_losses.append(avg_epoch_loss)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

print("训练完成！")

# --- 5. 评估模型 ---
model.eval()
test_loss = 0.0
all_predictions = []
all_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        # 数据已预先移动到GPU，这里无需再次移动
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item() * inputs.size(0)

        # 将预测结果和真实值从 GPU 移动回 CPU 进行处理和可视化
        all_predictions.extend(outputs.squeeze().cpu().tolist())
        all_targets.extend(targets.squeeze().cpu().tolist())

avg_test_loss = test_loss / len(test_dataset)
print(f"\n测试集 MSE Loss: {avg_test_loss:.4f}")

# --- 6. 结果可视化 ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses)
plt.title("训练损失曲线")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(all_targets, all_predictions, alpha=0.7)
plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--', label='理想预测')
plt.title("测试集预测值 vs 真实值")
plt.xlabel("真实房价")
plt.ylabel("预测房价")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

print("\n部分预测结果:")
for i in range(5):
    print(f"真实值: {all_targets[i]:.2f}, 预测值: {all_predictions[i]:.2f}")
