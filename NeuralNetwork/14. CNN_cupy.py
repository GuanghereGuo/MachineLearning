import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sklearn.datasets
from abc import ABC, abstractmethod
from collections import OrderedDict

# ===================================================================
# --- 1. GPU/CPU 后端切换 ---
# 设置为 True 来使用 CuPy (GPU)，设置为 False 来使用 NumPy (CPU)
USE_GPU = True

if USE_GPU:
    try:
        import cupy as cp

        xp = cp
        print("Using CuPy for GPU acceleration.")
    except ImportError:
        print("CuPy not found. Falling back to NumPy.")
        xp = np
else:
    xp = np
    print("Using NumPy for CPU computation.")
# ===================================================================


# ===================================================================
# 工具函数 (已修复 im2col)
# ===================================================================
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # --- 关键修复：修改 xp.pad 的参数格式 ---
    img = xp.pad(input_data, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")

    col = xp.zeros((N, C, filter_h, filter_w, out_h, out_w), dtype=input_data.dtype)

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

    img = xp.zeros(
        (N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1), dtype=col.dtype
    )
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x]

    return img[:, :, pad : H + pad, pad : W + pad]


def softmax(x):
    if x.ndim == 2:
        x = x - xp.max(x, axis=1, keepdims=True)
        return xp.exp(x) / xp.sum(xp.exp(x), axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - xp.max(x)
        return xp.exp(x) / xp.sum(xp.exp(x))
    else:
        raise ValueError("Input must be a 1D or 2D array.")


def relu(x):
    return xp.maximum(0, x)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if t.ndim == 1:
        t = t.reshape(1, -1)

    batch_size = y.shape[0]
    return -xp.sum(t * xp.log(y + 1e-7)) / batch_size


# ===================================================================
# 层定义 (已适配 xp)
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
        return dout


class Affine(Layer):
    def __init__(self, W, b):
        self.W, self.b = W, b
        self.x, self.original_x_shape = None, None
        self.dW, self.db = None, None

    def forward(self, x):
        self.original_x_shape, self.x = x.shape, x
        return xp.dot(self.x, self.W) + self.b

    def backward(self, dout):
        dx = xp.dot(dout, self.W.T)
        self.dW = xp.dot(self.x.T, dout)
        self.db = xp.sum(dout, axis=0)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss, self.y, self.t = None, None, None

    def forward(self, x, t):
        self.t, self.y = t, softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        return (self.y - self.t) / batch_size


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W, self.b = W, b
        self.stride, self.pad = stride, pad
        self.x, self.col, self.col_W = None, None, None
        self.dW, self.db = None, None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)
        self.col = im2col(x, FH, FW, self.stride, self.pad)
        self.col_W = self.W.reshape(FN, -1).T
        out = xp.dot(self.col, self.col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        self.x = x
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)
        self.db = xp.sum(dout, axis=0)
        self.dW = xp.dot(self.col.T, dout).transpose(1, 0).reshape(FN, C, FH, FW)
        dcol = xp.dot(dout, self.col_W.T)
        return col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)


class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h, self.pool_w = pool_h, pool_w
        self.stride, self.pad = stride, pad
        self.x, self.arg_max = None, None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad).reshape(
            -1, self.pool_h * self.pool_w
        )
        self.arg_max = xp.argmax(col, axis=1)
        out = xp.max(col, axis=1).reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        self.x = x
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.pool_h * self.pool_w
        dmax = xp.zeros((dout.size, pool_size), dtype=dout.dtype)
        dmax[xp.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dcol = dmax.reshape(dout.shape + (pool_size,)).reshape(
            dout.shape[0] * dout.shape[1] * dout.shape[2], -1
        )
        return col2im(
            dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad
        )


class Flatten(Layer):
    def __init__(self):
        self.original_shape = None

    def forward(self, x):
        self.original_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.original_shape)


# ===================================================================
# 优化器 (已适配 xp)
# ===================================================================
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
                self.m[key] = xp.zeros_like(params[key])
                self.v[key] = xp.zeros_like(params[key])
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (
                grads[key] ** 2
            )
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            params[key] -= self.lr * m_hat / (xp.sqrt(v_hat) + self.epsilon)


# ===================================================================
# CNN 网络 (已适配 xp)
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

        C, H, W = input_dim
        FN1, FS1, P1, S1 = (
            conv_param_1["filter_num"],
            conv_param_1["filter_size"],
            conv_param_1["pad"],
            conv_param_1["stride"],
        )
        conv1_out_h = (H + 2 * P1 - FS1) // S1 + 1
        pool1_out_h = conv1_out_h // 2

        FN2, FS2, P2, S2 = (
            conv_param_2["filter_num"],
            conv_param_2["filter_size"],
            conv_param_2["pad"],
            conv_param_2["stride"],
        )
        conv2_out_h = (pool1_out_h + 2 * P2 - FS2) // S2 + 1
        pool2_out_h = conv2_out_h // 2
        flattened_size = FN2 * pool2_out_h * pool2_out_h

        self.params = {}
        self.params["W1"] = xp.random.randn(FN1, C, FS1, FS1) * xp.sqrt(
            2.0 / (C * FS1 * FS1)
        )
        self.params["b1"] = xp.zeros(FN1)
        self.params["W2"] = xp.random.randn(FN2, FN1, FS2, FS2) * xp.sqrt(
            2.0 / (FN1 * FS2 * FS2)
        )
        self.params["b2"] = xp.zeros(FN2)
        self.params["W3"] = xp.random.randn(flattened_size, hidden_size) * xp.sqrt(
            2.0 / flattened_size
        )
        self.params["b3"] = xp.zeros(hidden_size)
        self.params["W4"] = xp.random.randn(hidden_size, output_size) * xp.sqrt(
            2.0 / hidden_size
        )
        self.params["b4"] = xp.zeros(output_size)

        self.layers = OrderedDict(
            [
                ("Conv1", Convolution(self.params["W1"], self.params["b1"], S1, P1)),
                ("Relu1", ReLu()),
                ("Pool1", Pooling(pool_h=2, pool_w=2, stride=2)),
                ("Conv2", Convolution(self.params["W2"], self.params["b2"], S2, P2)),
                ("Relu2", ReLu()),
                ("Pool2", Pooling(pool_h=2, pool_w=2, stride=2)),
                ("Flatten", Flatten()),
                ("Affine1", Affine(self.params["W3"], self.params["b3"])),
                ("Relu3", ReLu()),
                ("Affine2", Affine(self.params["W4"], self.params["b4"])),
            ]
        )

        self.loss_layer = SoftmaxWithLoss()

    def forward(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        return self.loss_layer.forward(self.forward(x), t)

    def gradient(self, x, t):
        self.loss(x, t)
        dout = self.loss_layer.backward(1)
        layers = reversed(list(self.layers.values()))
        for layer in layers:
            dout = layer.backward(dout)
        grads = {
            "W1": self.layers["Conv1"].dW,
            "b1": self.layers["Conv1"].db,
            "W2": self.layers["Conv2"].dW,
            "b2": self.layers["Conv2"].db,
            "W3": self.layers["Affine1"].dW,
            "b3": self.layers["Affine1"].db,
            "W4": self.layers["Affine2"].dW,
            "b4": self.layers["Affine2"].db,
        }
        return grads

    def predict(self, x):
        return xp.argmax(self.forward(x), axis=1)

    def accuracy(self, x, t):
        y = self.predict(x)
        if t.ndim != 1:
            t = xp.argmax(t, axis=1)
        return xp.sum(y == t) / float(x.shape[0])


# ===================================================================
# 主训练循环 (已适配 xp)
# ===================================================================
if __name__ == "__main__":
    mnist = sklearn.datasets.fetch_openml(
        "mnist_784", version=1, as_frame=False, parser="auto"
    )
    data, target = mnist.data, mnist.target

    x_train, t_train = data[:60000], target[:60000].astype(np.int32)
    x_test, t_test = data[60000:], target[60000:].astype(np.int32)

    x_train, x_test = x_train / 255.0, x_test / 255.0
    t_train, t_test = np.eye(10)[t_train], np.eye(10)[t_test]
    x_train, x_test = x_train.reshape(-1, 1, 28, 28), x_test.reshape(-1, 1, 28, 28)

    if USE_GPU:
        x_train, t_train = xp.asarray(x_train), xp.asarray(t_train)
        x_test, t_test = xp.asarray(x_test), xp.asarray(t_test)

    epoch_num, batch_size = 10, 100
    network, optimizer = CNN(), Adam()

    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size // batch_size, 1)

    train_loss_list, train_acc_list, test_acc_list = [], [], []

    for epoch in range(epoch_num):
        idx = xp.random.permutation(train_size)
        x_train_shuffled, t_train_shuffled = x_train[idx], t_train[idx]

        for i in tqdm(range(iter_per_epoch), desc=f"Epoch {epoch+1}/{epoch_num}"):
            batch_x = x_train_shuffled[i * batch_size : (i + 1) * batch_size]
            batch_t = t_train_shuffled[i * batch_size : (i + 1) * batch_size]

            grads = network.gradient(batch_x, batch_t)
            optimizer.update(network.params, grads)

            loss = network.loss_layer.loss.item()
            train_loss_list.append(loss)

        train_acc = network.accuracy(x_train[:5000], t_train[:5000])
        test_acc = network.accuracy(x_test, t_test)

        train_acc_list.append(train_acc.item())
        test_acc_list.append(test_acc.item())

        print(
            f"Epoch {epoch + 1}/{epoch_num}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
        )

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label="Train Loss")
    plt.xlabel("Iteration"), plt.ylabel("Loss"), plt.title("Training Loss")
    plt.legend(), plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label="Train Accuracy")
    plt.plot(test_acc_list, label="Test Accuracy")
    plt.xlabel("Epoch"), plt.ylabel("Accuracy"), plt.title("Accuracy")
    plt.legend(), plt.grid(True)

    plt.show()
    print("Training complete.")
    print(
        f"Final Train Accuracy: {train_acc_list[-1]:.4f}, Final Test Accuracy: {test_acc_list[-1]:.4f}"
    )
