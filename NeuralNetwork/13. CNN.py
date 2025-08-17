import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sklearn.datasets
from abc import ABC, abstractmethod
from collections import OrderedDict


# ===================================================================
# 工具函数 (无修改)
# ===================================================================
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, ((0,), (0,), (pad,), (pad,)), mode="constant")
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    return col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(
        0, 3, 4, 5, 1, 2
    )

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x]

    return img[:, :, pad : H + pad, pad : W + pad]


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


# ===================================================================
# 层定义 (新增 Pooling 和 Flatten)
# ===================================================================
class Layer(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, dout):
        pass


class ReLu(Layer):
    def __init__(self):
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
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        self.x = None
        self.col = None
        self.col_W = None
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        self.col = im2col(x, FH, FW, self.stride, self.pad)
        self.col_W = self.W.reshape(FN, -1).T

        out = np.dot(self.col, self.col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        return dx


# --- 新增 Pooling 层 ---
class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        self.arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx


# --- 新增 Flatten 层 ---
class Flatten(Layer):
    def __init__(self):
        self.original_shape = None

    def forward(self, x):
        self.original_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.original_shape)


# ===================================================================
# 优化器 (无修改)
# ===================================================================
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
            if key not in self.v:
                self.v[key] = np.zeros_like(params[key])
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (
                grads[key] ** 2
            )
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


# ===================================================================
# CNN 网络 (重大修改)
# ===================================================================
class CNN:
    def __init__(
        self,
        input_dim=(1, 28, 28),
        conv_param_1={"filter_num": 32, "filter_size": 3, "pad": 1, "stride": 1},
        conv_param_2={"filter_num": 64, "filter_size": 3, "pad": 1, "stride": 1},
        hidden_size=100,
        output_size=10,
    ):

        # --- 1. 计算卷积/池化层输出尺寸和全连接层输入尺寸 ---
        # 这是一个关键步骤，确保网络各层尺寸匹配
        C, H, W = input_dim
        FN1, FS1, P1, S1 = (
            conv_param_1["filter_num"],
            conv_param_1["filter_size"],
            conv_param_1["pad"],
            conv_param_1["stride"],
        )
        conv1_out_h = (H + 2 * P1 - FS1) // S1 + 1
        conv1_out_w = (W + 2 * P1 - FS1) // S1 + 1
        pool1_out_h = conv1_out_h // 2
        pool1_out_w = conv1_out_w // 2

        FN2, FS2, P2, S2 = (
            conv_param_2["filter_num"],
            conv_param_2["filter_size"],
            conv_param_2["pad"],
            conv_param_2["stride"],
        )
        conv2_out_h = (pool1_out_h + 2 * P2 - FS2) // S2 + 1
        conv2_out_w = (pool1_out_w + 2 * P2 - FS2) // S2 + 1
        pool2_out_h = conv2_out_h // 2
        pool2_out_w = conv2_out_w // 2

        # 展平后的尺寸
        flattened_size = FN2 * pool2_out_h * pool2_out_w

        # --- 2. 初始化参数 ---
        self.params = {}
        # He 初始化，更适合ReLU
        self.params["W1"] = np.random.randn(FN1, C, FS1, FS1) * np.sqrt(
            2.0 / (C * FS1 * FS1)
        )
        self.params["b1"] = np.zeros(FN1)
        self.params["W2"] = np.random.randn(FN2, FN1, FS2, FS2) * np.sqrt(
            2.0 / (FN1 * FS2 * FS2)
        )
        self.params["b2"] = np.zeros(FN2)
        self.params["W3"] = np.random.randn(flattened_size, hidden_size) * np.sqrt(
            2.0 / flattened_size
        )
        self.params["b3"] = np.zeros(hidden_size)
        self.params["W4"] = np.random.randn(hidden_size, output_size) * np.sqrt(
            2.0 / hidden_size
        )
        self.params["b4"] = np.zeros(output_size)

        # --- 3. 构建网络层 ---
        # 使用 OrderedDict 来方便地命名和访问层
        self.layers = OrderedDict()
        self.layers["Conv1"] = Convolution(self.params["W1"], self.params["b1"], S1, P1)
        self.layers["Relu1"] = ReLu()
        self.layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers["Conv2"] = Convolution(self.params["W2"], self.params["b2"], S2, P2)
        self.layers["Relu2"] = ReLu()
        self.layers["Pool2"] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers["Flatten"] = Flatten()
        self.layers["Affine1"] = Affine(self.params["W3"], self.params["b3"])
        self.layers["Relu3"] = ReLu()
        self.layers["Affine2"] = Affine(self.params["W4"], self.params["b4"])

        self.loss_layer = SoftmaxWithLoss()

    def forward(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.forward(x)
        return self.loss_layer.forward(y, t)

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.loss_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 收集梯度
        grads = {}
        grads["W1"], grads["b1"] = self.layers["Conv1"].dW, self.layers["Conv1"].db
        grads["W2"], grads["b2"] = self.layers["Conv2"].dW, self.layers["Conv2"].db
        grads["W3"], grads["b3"] = self.layers["Affine1"].dW, self.layers["Affine1"].db
        grads["W4"], grads["b4"] = self.layers["Affine2"].dW, self.layers["Affine2"].db

        return grads

    def predict(self, x):
        y = self.forward(x)
        return np.argmax(y, axis=1)

    def accuracy(self, x, t):
        y = self.predict(x)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


# ===================================================================
# 主训练循环 (重大修改)
# ===================================================================
if __name__ == "__main__":
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

    # 归一化
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # One-Hot 编码
    t_train = np.eye(10)[t_train]
    t_test = np.eye(10)[t_test]

    # --- 关键修改: 将数据 reshape 为 4D 张量 ---
    x_train = x_train.reshape(-1, 1, 28, 28)
    x_test = x_test.reshape(-1, 1, 28, 28)

    epoch_num = 10  # 减少 epoch 数量，因为 Adam 收敛快
    batch_size = 100

    # 创建神经网络
    network = CNN()
    optimizer = Adam()

    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size // batch_size, 1)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for epoch in range(epoch_num):
        idx = np.random.permutation(train_size)
        x_train_shuffled = x_train[idx]
        t_train_shuffled = t_train[idx]

        for i in tqdm(range(iter_per_epoch), desc=f"Epoch {epoch+1}/{epoch_num}"):
            batch_x = x_train_shuffled[i * batch_size : (i + 1) * batch_size]
            batch_t = t_train_shuffled[i * batch_size : (i + 1) * batch_size]

            # 计算梯度 (注意：loss的计算已包含在gradient方法内部)
            grads = network.gradient(batch_x, batch_t)

            # 更新参数
            optimizer.update(network.params, grads)

            # 记录损失
            loss = network.loss_layer.loss
            train_loss_list.append(loss)

        # 每个 epoch 结束后计算准确率
        # 注意：为节省时间，可以从训练集中抽样一小部分来计算训练准确率
        train_acc = network.accuracy(x_train[:5000], t_train[:5000])  # 抽样计算
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(
            f"Epoch {epoch + 1}/{epoch_num}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
        )

    # 绘制损失和准确率曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label="Train Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label="Train Accuracy")
    plt.plot(test_acc_list, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()
    print("Training complete.")
    print(
        f"Final Train Accuracy: {train_acc_list[-1]:.4f}, Final Test Accuracy: {test_acc_list[-1]:.4f}"
    )
