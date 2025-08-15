import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from tqdm import tqdm
import time
from collections import OrderedDict  # 引入有序字典，保证网络层次的顺序


class Relu:
    """ReLU 激活函数层"""

    def __init__(self):
        self.mask = None

    def forward(self, x):
        # 在前向传播时，记录下小于等于0的元素位置
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        # 在反向传播时，对于前向传播中被抑制的神经元（值为0），其梯度也为0
        dout[self.mask] = 0
        dx = dout
        return dx


class Affine:
    """全连接层 (Wx + b)"""

    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        # 记录输入 x，为反向传播计算 dW 做准备
        self.x = x
        out = cp.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        # 根据链式法则计算梯度
        dx = cp.dot(dout, self.W.T)
        self.dW = cp.dot(self.x.T, dout)
        self.db = cp.sum(dout, axis=0)
        return dx


class SoftmaxWithLoss:
    """Softmax 和 Cross-Entropy-Error 的组合层"""

    def __init__(self):
        self.loss = None
        self.y = None  # softmax的输出
        self.t = None  # 监督数据(one-hot vector)

    def forward(self, x, t):
        self.t = t
        # 使用稳定的 softmax 实现
        if x.ndim == 2:
            x_shifted = x - cp.max(x, axis=1, keepdims=True)
            self.y = cp.exp(x_shifted) / cp.sum(
                cp.exp(x_shifted), axis=1, keepdims=True
            )
        else:  # 处理一维情况
            x_shifted = x - cp.max(x)
            self.y = cp.exp(x_shifted) / cp.sum(cp.exp(x_shifted))

        # 计算损失
        batch_size = self.y.shape[0]
        self.loss = -cp.sum(self.t * cp.log(self.y + 1e-7)) / batch_size
        return self.loss

    def backward(self, dout=1):
        # 反向传播的初始梯度
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


# --- NLayerNet 类 (核心泛化) ---
class NLayerNet:
    """
    一个通用的N层全连接神经网络

    Parameters
    ----------
    input_size : 输入大小 (MNIST: 784)
    hidden_size_list : 隐藏层神经元数量的列表 (e.g., [100, 50, 20])
    output_size : 输出大小 (MNIST: 10)
    weight_init_std : 权重初始化标准差
    """

    def __init__(self, input_size, hidden_size_list, output_size, weight_init_std=0.01):
        # 1. 初始化权重和偏置
        self.params = {}
        all_layer_sizes = [input_size] + hidden_size_list + [output_size]
        for i in range(1, len(all_layer_sizes)):
            # He 初始化对于 ReLU 更优，但我们遵循原代码使用标准差
            # scale = cp.sqrt(2.0 / all_layer_sizes[i-1]) # He initialization
            scale = weight_init_std
            self.params[f"W{i}"] = scale * cp.random.randn(
                all_layer_sizes[i - 1], all_layer_sizes[i], dtype=cp.float32
            )
            self.params[f"b{i}"] = cp.zeros(all_layer_sizes[i], dtype=cp.float32)

        # 2. 构建网络层
        # 使用 OrderedDict 来保证层的顺序
        self.layers = OrderedDict()
        # 循环添加 Affine 和 Relu 层
        for i in range(1, len(hidden_size_list) + 1):
            self.layers[f"Affine{i}"] = Affine(
                self.params[f"W{i}"], self.params[f"b{i}"]
            )
            self.layers[f"Relu{i}"] = Relu()

        # 添加最后一层 (输出层)
        last_layer_idx = len(hidden_size_list) + 1
        self.layers[f"Affine{last_layer_idx}"] = Affine(
            self.params[f"W{last_layer_idx}"], self.params[f"b{last_layer_idx}"]
        )

        # 最后的损失层
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        # 前向传播：按顺序通过所有层
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        # 计算损失：前向传播到最后，然后通过损失层
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = cp.argmax(y, axis=1)
        if t.ndim != 1:
            t = cp.argmax(t, axis=1)
        accuracy = cp.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # 1. 前向传播
        self.loss(x, t)

        # 2. 反向传播
        # 从损失层开始，获取初始梯度
        dout = self.last_layer.backward(1)

        # 反向遍历所有层（除了损失层）
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 3. 收集梯度
        # 从每个 Affine 层中提取计算好的梯度
        grads = {}
        num_hidden_layers = len(self.layers) // 2
        for i in range(1, num_hidden_layers + 2):
            grads[f"W{i}"] = self.layers[f"Affine{i}"].dW
            grads[f"b{i}"] = self.layers[f"Affine{i}"].db

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
    epoch_num = 200
    batch_size = 1000
    learning_rate = 0.1

    train_size = x_train.shape[0]
    iter_per_epoch = train_size // batch_size

    # --- 初始化网络和记录列表 ---
    # 实例化一个4层网络 (输入层 -> 隐藏层1 -> 隐藏层2 -> 输出层)
    # 这里的 n=4 指的是层的总数，对应2个隐藏层
    print("Initializing a 4-layer network (2 hidden layers)...")
    net = NLayerNet(input_size=28 * 28, hidden_size_list=[512, 64, 256], output_size=10)

    epoch_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # --- 训练过程 ---
    cp.cuda.Stream.null.synchronize()
    start_time = time.time()

    for epoch in range(epoch_num):
        indices = cp.random.permutation(train_size)
        epoch_loss = 0.0

        print(f"Epoch {epoch + 1}/{epoch_num}")
        for i in tqdm(range(iter_per_epoch)):
            batch_indices = indices[i * batch_size : (i + 1) * batch_size]
            x_batch = x_train[batch_indices]
            t_batch = t_train[batch_indices]

            # 计算梯度并更新权重
            grads = net.gradient(x_batch, t_batch)
            for key in net.params.keys():  # 注意这里是 net.params
                net.params[key] -= learning_rate * grads[key]

            # 累加损失值
            loss = net.loss(x_batch, t_batch)  # loss()方法现在返回一个cupy标量
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / iter_per_epoch
        epoch_loss_list.append(avg_epoch_loss)

        train_acc = net.accuracy(x_train, t_train)
        test_acc = net.accuracy(x_test, t_test)

        train_acc_list.append(train_acc.item())
        test_acc_list.append(test_acc.item())

        print(
            f"Epoch {epoch+1} -> Avg Loss: {avg_epoch_loss:.4f}, Train Acc: {train_acc.item():.4f}, Test Acc: {test_acc.item():.4f}"
        )

    cp.cuda.Stream.null.synchronize()
    end_time = time.time()
    print(f"\nTotal training time: {end_time - start_time:.2f} seconds")

    # --- 绘制图表 ---
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch_num + 1), epoch_loss_list, marker="o", label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss per Epoch")
    plt.xticks(range(1, epoch_num + 1))
    plt.grid(True)
    plt.legend()

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
    plt.ylim(0.8, 1.0)
    plt.xticks(range(1, epoch_num + 1))
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"Final Train Accuracy: {train_acc_list[-1]:.4f}")
    print(f"Final Test Accuracy: {test_acc_list[-1]:.4f}")
    print("Training complete.")
