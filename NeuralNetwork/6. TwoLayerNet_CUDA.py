import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from tqdm import tqdm
import time  # 引入 time 模块用于计时


# --- 激活函数和损失函数 (已正确使用 CuPy, 无需改动) ---
def softmax(x):
    if x.ndim == 2:
        x = x - cp.max(x, axis=1, keepdims=True)
        return cp.exp(x) / cp.sum(cp.exp(x), axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - cp.max(x)
        return cp.exp(x) / cp.sum(cp.exp(x))
    else:
        raise ValueError("Input must be a 1D or 2D array.")


def relu(x):
    return cp.maximum(0, x)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if t.ndim == 1:
        t = t.reshape(1, -1)
    batch_size = y.shape[0]
    return -cp.sum(t * cp.log(y + 1e-7)) / batch_size


# --- TwoLayerNet 类 (已正确使用 CuPy, 无需改动) ---
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.prams = {}
        self.prams["W1"] = weight_init_std * cp.random.randn(
            input_size, hidden_size, dtype=cp.float32
        )
        self.prams["b1"] = cp.zeros(hidden_size, dtype=cp.float32)
        self.prams["W2"] = weight_init_std * cp.random.randn(
            hidden_size, output_size, dtype=cp.float32
        )
        self.prams["b2"] = cp.zeros(output_size, dtype=cp.float32)

    def predict(self, x):
        W1, b1 = self.prams["W1"], self.prams["b1"]
        W2, b2 = self.prams["W2"], self.prams["b2"]
        a1 = cp.dot(x, W1) + b1
        z1 = relu(a1)
        a2 = cp.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = cp.argmax(y, axis=1)
        if t.ndim != 1:
            t = cp.argmax(t, axis=1)
        accuracy = cp.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        W1, b1 = self.prams["W1"], self.prams["b1"]
        W2, b2 = self.prams["W2"], self.prams["b2"]
        batch_num = x.shape[0]
        a1 = cp.dot(x, W1) + b1
        z1 = relu(a1)
        a2 = cp.dot(z1, W2) + b2
        y = softmax(a2)
        dy = (y - t) / batch_num
        grads = {"W2": cp.dot(z1.T, dy), "b2": cp.sum(dy, axis=0)}
        da1 = cp.dot(dy, W2.T)
        dz1 = da1 * (z1 > 0)
        grads["W1"] = cp.dot(x.T, dz1)
        grads["b1"] = cp.sum(dz1, axis=0)
        return grads


if __name__ == "__main__":
    # --- 数据获取与预处理 (在CPU上完成) ---
    print("Fetching MNIST dataset...")
    mnist = sklearn.datasets.fetch_openml(
        "mnist_784", version=1, as_frame=False, parser="liac-arff"
    )
    data = mnist.data
    target = mnist.target
    print("Dataset loaded.")

    x_train_np = data[:60000] / 255.0
    t_train_np = np.eye(10)[target[:60000].astype(np.int32)]
    x_test_np = data[60000:] / 255.0
    t_test_np = np.eye(10)[target[60000:].astype(np.int32)]

    # --- 数据一次性转移到 GPU ---
    print("Moving data to GPU...")
    x_train = cp.asarray(x_train_np, dtype=cp.float32)
    t_train = cp.asarray(t_train_np, dtype=cp.float32)
    x_test = cp.asarray(x_test_np, dtype=cp.float32)
    t_test = cp.asarray(t_test_np, dtype=cp.float32)
    print("Data ready on GPU.")

    # --- 超参数设置 ---
    epoch_num = 20
    batch_size = 100
    learning_rate = 0.1
    train_size = x_train.shape[0]
    iter_per_epoch = train_size // batch_size

    # --- 初始化网络和记录列表 ---
    net = TwoLayerNet(input_size=28 * 28, hidden_size=10000, output_size=10)

    # --- 优化点 1: 仅记录每个 epoch 的平均损失和准确率 ---
    epoch_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # --- 训练过程 ---
    # 添加 GPU 同步以准确计时
    cp.cuda.Stream.null.synchronize()
    start_time = time.time()

    for epoch in range(epoch_num):
        # --- 优化点 2: 直接在 GPU 上生成和操作索引，避免循环内 CPU->GPU 传输 ---
        indices = cp.random.permutation(train_size)

        epoch_loss = 0.0  # 用于累加当前 epoch 的总损失

        print(f"Epoch {epoch + 1}/{epoch_num}")
        for i in tqdm(range(iter_per_epoch)):
            # 直接在 GPU 上进行切片，无数据传输开销
            batch_indices = indices[i * batch_size : (i + 1) * batch_size]
            x_batch = x_train[batch_indices]
            t_batch = t_train[batch_indices]

            # 计算梯度并更新权重 (纯 GPU 操作)
            grads = net.gradient(x_batch, t_batch)
            for key in net.prams.keys():
                net.prams[key] -= learning_rate * grads[key]

            # 累加损失值
            loss = net.loss(x_batch, t_batch)
            epoch_loss += loss.item()

        # 计算并记录当前 epoch 的平均损失
        avg_epoch_loss = epoch_loss / iter_per_epoch
        epoch_loss_list.append(avg_epoch_loss)

        # 计算准确率 (纯 GPU 操作)
        train_acc = net.accuracy(x_train, t_train)
        test_acc = net.accuracy(x_test, t_test)

        # 将结果从 GPU 取回 CPU 用于记录和打印
        train_acc_list.append(train_acc.item())
        test_acc_list.append(test_acc.item())

        print(
            f"Epoch {epoch+1} -> Avg Loss: {avg_epoch_loss:.4f}, Train Acc: {train_acc.item():.4f}, Test Acc: {test_acc.item():.4f}"
        )

    # 添加 GPU 同步以准确计时
    cp.cuda.Stream.null.synchronize()
    end_time = time.time()
    print(f"\nTotal training time: {end_time - start_time:.2f} seconds")

    # --- 优化点 3: 绘制清晰的、基于 Epoch 的图表 ---
    plt.figure(figsize=(14, 6))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch_num + 1), epoch_loss_list, marker="o", label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss per Epoch")
    plt.xticks(range(1, epoch_num + 1))
    plt.grid(True)
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(
        range(1, epoch_num + 1), train_acc_list, marker="o", label="Train Accuracy"
    )
    plt.plot(
        range(1, epoch_num + 1),
        test_acc_list,
        marker="s",
        linestyle="--",
        label="Test Accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Epoch")
    plt.ylim(0.8, 1.0)  # 设置合理的Y轴范围以突出变化
    plt.xticks(range(1, epoch_num + 1))
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"Final Train Accuracy: {train_acc_list[-1]:.4f}")
    print(f"Final Test Accuracy: {test_acc_list[-1]:.4f}")
    print("Training complete.")
