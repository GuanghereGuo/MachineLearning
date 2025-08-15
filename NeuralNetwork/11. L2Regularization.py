import numpy as np
import matplotlib.pyplot as plt
import sklearn
from abc import ABC, abstractmethod
from tqdm import tqdm

# -------------------------------------------------------------------
# 辅助函数 (无修改)
# -------------------------------------------------------------------


def softmax(x):
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))
    else:
        raise ValueError("Input must be a 1D or 2D array.")


def relu(x):
    return np.maximum(0, x)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if t.ndim == 1:
        t = t.reshape(1, -1)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


# -------------------------------------------------------------------
# 层定义
# -------------------------------------------------------------------


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
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
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
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class BatchNormalization(Layer):
    # 接收外部传入的 gamma 和 beta 对象
    def __init__(self, gamma, beta, eps=1e-5):
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        self.x_normalized = None
        self.std = None
        self.mean = None
        self.x = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x):
        self.x = x
        self.mean = x.mean(axis=0)
        self.var = x.var(axis=0)
        self.std = np.sqrt(self.var + self.eps)
        self.x_normalized = (x - self.mean) / self.std
        out = self.gamma * self.x_normalized + self.beta
        return out

    def backward(self, dout):
        N, D = dout.shape
        self.dbeta = np.sum(dout, axis=0)
        self.dgamma = np.sum(self.x_normalized * dout, axis=0)
        dx_normalized = dout * self.gamma
        dvar = (
            np.sum(dx_normalized * (self.x - self.mean), axis=0) * -0.5 * (self.std**-3)
        )
        dmean = np.sum(dx_normalized * -1 / self.std, axis=0) + dvar * np.mean(
            -2 * (self.x - self.mean), axis=0
        )
        dx = (
            (dx_normalized / self.std)
            + (dvar * 2 * (self.x - self.mean) / N)
            + (dmean / N)
        )
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        super().__init__()
        self.y = None
        self.t = None
        self.loss = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


# -------------------------------------------------------------------
# 神经网络定义 (整合了BN学习和L2正则化)
# -------------------------------------------------------------------


class TwoLayerNetWithLayers:
    # 【L2正则化修改点 1】: 在构造函数中添加 weight_decay_lambda 参数
    def __init__(self, input_size, hidden_size, output_size, weight_decay_lambda=0.0):
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        # He 初始化
        self.params["W1"] = np.random.randn(input_size, hidden_size) * np.sqrt(
            2.0 / input_size
        )
        self.params["b1"] = np.zeros(hidden_size)
        # 【BN学习修改点】: 将 gamma 和 beta 添加到参数字典中
        self.params["gamma1"] = np.ones(hidden_size)
        self.params["beta1"] = np.zeros(hidden_size)

        self.params["W2"] = np.random.randn(hidden_size, output_size) * np.sqrt(
            2.0 / hidden_size
        )
        self.params["b2"] = np.zeros(output_size)
        self.params["gamma2"] = np.ones(output_size)
        self.params["beta2"] = np.zeros(output_size)

        # 【BN学习修改点】: 创建层时，传入参数对象
        self.layers = [
            Affine(self.params["W1"], self.params["b1"]),
            BatchNormalization(self.params["gamma1"], self.params["beta1"]),
            ReLu(),
            Affine(self.params["W2"], self.params["b2"]),
            BatchNormalization(self.params["gamma2"], self.params["beta2"]),
        ]
        self.loss_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    # 【L2正则化修改点 2】: 修改 loss 方法，使其包含 L2 正则化项
    def loss(self, x, t):
        y = self.predict(x)
        data_loss = self.loss_layer.forward(y, t)

        weight_decay = 0
        weight_decay += 0.5 * self.weight_decay_lambda * np.sum(self.params["W1"] ** 2)
        weight_decay += 0.5 * self.weight_decay_lambda * np.sum(self.params["W2"] ** 2)

        return data_loss + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 【整合修改点】: 修改 gradient 方法，收集所有梯度并添加正则化项
    def gradient(self, x, t):
        # 1. 前向传播 (为反向传播准备好各层中间变量)
        self.loss(x, t)

        # 2. 反向传播 (计算数据损失的梯度)
        dout = 1
        dout = self.loss_layer.backward(dout)

        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        # 3. 收集所有参数的梯度
        grads = {
            "W1": self.layers[0].dW,
            "b1": self.layers[0].db,
            "gamma1": self.layers[1].dgamma,
            "beta1": self.layers[1].dbeta,
            "W2": self.layers[3].dW,
            "b2": self.layers[3].db,
            "gamma2": self.layers[4].dgamma,
            "beta2": self.layers[4].dbeta,
        }

        # 4. 【L2正则化修改点 3】: 将正则化项的梯度添加到权重梯度上
        grads["W1"] += self.weight_decay_lambda * self.params["W1"]
        grads["W2"] += self.weight_decay_lambda * self.params["W2"]

        return grads


# -------------------------------------------------------------------
# 优化器 (无修改)
# -------------------------------------------------------------------


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}
        self.v = {}

    def update(self, params, grads):
        self.t += 1
        for key in params.keys():
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (
                grads[key] ** 2
            )
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


# -------------------------------------------------------------------
# 主训练流程
# -------------------------------------------------------------------

if __name__ == "__main__":
    # 加载数据
    mnist = sklearn.datasets.fetch_openml(
        "mnist_784", version=1, as_frame=False, parser="auto"
    )
    data = mnist.data
    target = mnist.target

    # 数据预处理
    x_train = data[:60000]
    t_train = target[:60000].astype(np.int32)
    x_test = data[60000:]
    t_test = target[60000:].astype(np.int32)

    x_train = x_train / 255.0
    x_test = x_test / 255.0
    t_train = np.eye(10)[t_train]
    t_test = np.eye(10)[t_test]

    # 设置超参数
    epoch_num = 10
    batch_size = 128
    hidden_size = 100
    learning_rate = 0.001
    # 【L2正则化修改点 4】: 设置正则化系数
    lambda_val = 0.05

    # 创建网络和优化器
    network = TwoLayerNetWithLayers(
        input_size=784,
        hidden_size=hidden_size,
        output_size=10,
        weight_decay_lambda=lambda_val,
    )
    optimizer = Adam(learning_rate=learning_rate)

    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size // batch_size, 1)

    # 记录训练过程
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    print(f"Starting training with L2 regularization (lambda={lambda_val})...")

    for epoch in range(epoch_num):
        # 每个epoch开始时打乱数据
        idx = np.random.permutation(train_size)
        x_train_shuffled = x_train[idx]
        t_train_shuffled = t_train[idx]

        for i in tqdm(range(iter_per_epoch), desc=f"Epoch {epoch+1}/{epoch_num}"):
            batch_x = x_train_shuffled[i * batch_size : (i + 1) * batch_size]
            batch_t = t_train_shuffled[i * batch_size : (i + 1) * batch_size]

            # 计算梯度 (已包含正则化项)
            grads = network.gradient(batch_x, batch_t)
            # 更新参数
            optimizer.update(network.params, grads)

            # 记录损失 (已包含正则化项)
            loss = network.loss(batch_x, batch_t)
            train_loss_list.append(loss)

        # 每个epoch结束后计算并记录准确率
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        last_loss = train_loss_list[-1]
        print(
            f"Epoch {epoch + 1}/{epoch_num} done. Loss: {last_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
        )

    # 绘制结果
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss (Data + Regularization)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label="Train Accuracy")
    plt.plot(test_acc_list, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("Training complete.")
    print(
        f"Final Train Accuracy: {train_acc_list[-1]:.4f}, Final Test Accuracy: {test_acc_list[-1]:.4f}"
    )
