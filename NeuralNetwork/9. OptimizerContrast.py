import numpy as np
import matplotlib.pyplot as plt
import sklearn
from abc import ABC, abstractmethod
from tqdm import tqdm

# ==============================================================================
#  您提供的所有类和函数定义（此处保持不变，无需修改）
# ==============================================================================


def softmax(x):
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))
    else:
        raise ValueError("Input must be a 1D or 2D array.")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if t.ndim == 1:
        t = t.reshape(1, -1)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
        it.iternext()
    return grad


class Layer(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, dout):
        pass


class ReLu(Layer):
    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, x):
        self.x = x.copy()
        return np.maximum(0, x)

    def backward(self, dout):
        dx = np.zeros_like(self.x)
        dx[self.x > 0] = dout[self.x > 0]
        return dx


class Affine(Layer):
    def __init__(self, W, b):
        super().__init__()
        self.x = None
        self.W = W
        self.b = b
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x.copy()
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        super().__init__()
        self.y = None
        self.t = None
        self.loss = None

    def forward(self, x, t):
        self.t = t.copy()
        self.y = softmax(x)
        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.y.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


class TwoLayerNetWithLayers:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        # self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        # self.params["b1"] = np.zeros(hidden_size)
        # self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        # self.params["b2"] = np.zeros(output_size)

        # apply He initialization for ReLU
        self.params["W1"] = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.params["b2"] = np.zeros(output_size)

        self.layers = [
            Affine(self.params["W1"], self.params["b1"]),
            ReLu(),
            Affine(self.params["W2"], self.params["b2"]),
        ]
        self.loss_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.loss_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        self.loss(x, t)  # 执行前向传播
        dout = self.loss_layer.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        grads = {
            "W1": self.layers[0].dW,
            "b1": self.layers[0].db,
            "W2": self.layers[2].dW,
            "b2": self.layers[2].db,
        }
        return grads


class SGD:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr, self.momentum, self.v = learning_rate, momentum, {}

    def update(self, params, grads):
        for key in params.keys():
            if key not in self.v:
                self.v[key] = np.zeros_like(params[key])
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    def __init__(self, learning_rate=0.01):
        self.lr, self.h = learning_rate, {}

    def update(self, params, grads):
        for key in params.keys():
            if key not in self.h:
                self.h[key] = np.zeros_like(params[key])
            self.h[key] += grads[key] ** 2
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class RMSProp:
    def __init__(self, learning_rate=0.001, decay_rate=0.99):
        self.lr, self.decay_rate, self.h = learning_rate, decay_rate, {}

    def update(self, params, grads):
        for key in params.keys():
            if key not in self.h:
                self.h[key] = np.zeros_like(params[key])
            self.h[key] = self.decay_rate * self.h[key] + (1 - self.decay_rate) * (
                grads[key] ** 2
            )
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.lr, self.beta1, self.beta2, self.epsilon = (
            learning_rate,
            beta1,
            beta2,
            epsilon,
        )
        self.t, self.m, self.v = 0, {}, {}

    def update(self, params, grads):
        self.t += 1
        for key in params.keys():
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
            if key not in self.v:
                self.v[key] = np.zeros_like(params[key])
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (
                grads[key] ** 2
            )
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


# ==============================================================================
#  主要逻辑修改部分
# ==============================================================================
if __name__ == "__main__":
    # 1. 数据加载与预处理 (只需执行一次)
    print("Loading MNIST dataset...")
    mnist = sklearn.datasets.fetch_openml(
        "mnist_784", version=1, as_frame=False, parser="auto"
    )
    data = mnist.data
    target = mnist.target

    x_train, x_test = data[:60000], data[60000:]
    t_train, t_test = target[:60000].astype(np.int32), target[60000:].astype(np.int32)

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # 将标签转换为 one-hot 编码
    t_train = np.eye(10)[t_train]
    t_test_one_hot = np.eye(10)[t_test]  # 保留原始标签用于 accuracy 计算
    print("Dataset loaded and preprocessed.")

    # 2. 初始化模型和优化器
    # 注意：为不同优化器设置了更常用的学习率
    optimizers = {
        "SGD": SGD(learning_rate=0.1),
        "Momentum": Momentum(learning_rate=0.05, momentum=0.9),
        "AdaGrad": AdaGrad(learning_rate=0.01),
        "RMSProp": RMSProp(learning_rate=0.001),
        "Adam": Adam(learning_rate=0.001),
    }

    networks = {}
    train_loss = {}
    train_acc = {}
    test_acc = {}

    for key in optimizers.keys():
        networks[key] = TwoLayerNetWithLayers(
            input_size=28 * 28, hidden_size=200, output_size=10
        )
        train_loss[key] = []
        train_acc[key] = []
        test_acc[key] = []

    # 3. 训练循环
    epoch_num = 20
    batch_size = 100
    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size // batch_size, 1)

    for epoch in range(epoch_num):
        print(f"\n--- Epoch {epoch + 1}/{epoch_num} ---")

        # 每个 epoch 开始时都打乱数据
        idx = np.random.permutation(train_size)
        x_train_shuffled = x_train[idx]
        t_train_shuffled = t_train[idx]

        for i in tqdm(range(iter_per_epoch), desc=f"Epoch {epoch+1} Training"):
            batch_x = x_train_shuffled[i * batch_size : (i + 1) * batch_size]
            batch_t = t_train_shuffled[i * batch_size : (i + 1) * batch_size]

            # 对每个模型进行训练
            for key, optimizer in optimizers.items():
                network = networks[key]
                grads = network.gradient(batch_x, batch_t)
                optimizer.update(network.params, grads)

        # 每个 epoch 结束后，计算并记录所有模型的损失和准确率
        print("Calculating accuracy for all models...")
        for key, network in networks.items():
            # 计算整个训练集上的损失和准确率
            loss = network.loss(x_train, t_train)
            tr_acc = network.accuracy(x_train, t_train)
            te_acc = network.accuracy(x_test, t_test_one_hot)

            train_loss[key].append(loss)
            train_acc[key].append(tr_acc)
            test_acc[key].append(te_acc)

            print(
                f"Optimizer: {key:<10} | Loss: {loss:.4f} | Train Acc: {tr_acc:.4f} | Test Acc: {te_acc:.4f}"
            )

    print("\nTraining complete.")

    # 4. 绘制损失和准确率曲线
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(18, 7))

    # 绘制损失函数曲线
    plt.subplot(1, 2, 1)
    for key, loss_list in train_loss.items():
        plt.plot(range(epoch_num), loss_list, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.ylim(0, 1.0)  # 限制y轴范围以便观察

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    for key, acc_list in test_acc.items():
        plt.plot(range(epoch_num), acc_list, label=key)

    # 如果想同时看训练集准确率，可以取消下面代码的注释
    # for key, acc_list in train_acc.items():
    #     plt.plot(range(epoch_num), acc_list, linestyle='--', label=f'{key} (Train)')

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy Comparison")
    plt.legend()
    plt.ylim(0.9, 1.0)  # 放大y轴范围以便观察后期差异

    plt.tight_layout()
    plt.show()
