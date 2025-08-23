# [CuPy] 导入 cupy 并将其简写为 cp
import cupy as cp
import matplotlib.pyplot as plt
import sklearn
from abc import ABC, abstractmethod
from tqdm import tqdm


# --- 辅助函数 (已修改为 CuPy) ---
def softmax(x):
    if x.ndim == 2:
        x = x - cp.max(x, axis=1, keepdims=True)
        # [CuPy] 使用 cp.exp 和 cp.sum
        return cp.exp(x) / cp.sum(cp.exp(x), axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - cp.max(x)
        return cp.exp(x) / cp.sum(cp.exp(x))
    else:
        raise ValueError("Input must be a 1D or 2D array.")


def sigmoid(x):
    # [CuPy] 使用 cp.exp
    return 1 / (1 + cp.exp(-x))


def relu(x):
    # [CuPy] 使用 cp.maximum
    return cp.maximum(0, x)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if t.ndim == 1:
        t = t.reshape(1, -1)
    batch_size = y.shape[0]
    # [CuPy] 使用 cp.sum 和 cp.log
    return -cp.sum(t * cp.log(y + 1e-7)) / batch_size


def numerical_gradient(f, x):
    h = 1e-4
    # [CuPy] 使用 cp.zeros_like
    grad = cp.zeros_like(x)
    # [CuPy] np.nditer 在 CuPy 中没有直接对应，但我们可以用索引迭代
    # 注意：这个函数在 CuPy 中会非常慢，因为涉及大量小的 GPU-CPU 数据交换。
    # 但由于我们主要使用反向传播，这里仅为保持 API 完整性而修改。
    it = cp.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx].item()  # .item() 从GPU获取单个值
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
        it.iternext()
    return grad


# --- 层定义 (已修改为 CuPy) ---
class Layer(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, dout):
        pass


class Dropout(Layer):
    def __init__(self, dropout_ratio=0.5):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            # [CuPy] 使用 cp.random.rand
            self.mask = cp.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


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
        # [CuPy] 使用 cp.dot
        out = cp.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        # [CuPy] 使用 cp.dot 和 cp.sum
        dx = cp.dot(dout, self.W.T)
        self.dW = cp.dot(self.x.T, dout)
        self.db = cp.sum(dout, axis=0)
        return dx


class BatchNormalization(Layer):
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
        # [CuPy] 所有操作自动在 CuPy 数组上进行
        self.mean = x.mean(axis=0)
        self.var = x.var(axis=0)
        self.std = cp.sqrt(self.var + self.eps)
        self.x_normalized = (x - self.mean) / self.std
        out = self.gamma * self.x_normalized + self.beta
        return out

    def backward(self, dout):
        N, D = dout.shape
        # [CuPy] cp.sum
        self.dbeta = cp.sum(dout, axis=0)
        self.dgamma = cp.sum(self.x_normalized * dout, axis=0)
        dx_normalized = dout * self.gamma
        dvar = (
            cp.sum(dx_normalized * (self.x - self.mean), axis=0) * -0.5 * (self.std**-3)
        )
        dmean = cp.sum(dx_normalized * -1 / self.std, axis=0) + dvar * cp.mean(
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


# --- 网络定义 (已修改为 CuPy) ---
class TwoLayerNetWithLayers:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dropout_ratio=0.5,
    ):
        self.params = {}
        # [CuPy] 使用 cp.random.randn, cp.sqrt, cp.zeros, cp.ones 初始化参数
        # 参数将直接在 GPU 上创建
        self.params["W1"] = cp.random.randn(input_size, hidden_size) * cp.sqrt(
            2.0 / input_size
        )
        self.params["b1"] = cp.zeros(hidden_size)
        self.params["gamma1"] = cp.ones(hidden_size)
        self.params["beta1"] = cp.zeros(hidden_size)
        self.params["W2"] = cp.random.randn(hidden_size, output_size) * cp.sqrt(
            2.0 / hidden_size
        )
        self.params["b2"] = cp.zeros(output_size)
        self.params["gamma2"] = cp.ones(output_size)
        self.params["beta2"] = cp.zeros(output_size)

        self.layers = [
            Affine(self.params["W1"], self.params["b1"]),
            BatchNormalization(self.params["gamma1"], self.params["beta1"]),
            ReLu(),
            Dropout(dropout_ratio),
            Affine(self.params["W2"], self.params["b2"]),
            BatchNormalization(self.params["gamma2"], self.params["beta2"]),
        ]
        self.loss_layer = SoftmaxWithLoss()
        self.dropout_layer = self.layers[3]

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=True):
        y = self.predict(x, train_flg)
        return self.loss_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x, train_flg=False)
        # [CuPy] 使用 cp.argmax
        y = cp.argmax(y, axis=1)
        if t.ndim != 1:
            t = cp.argmax(t, axis=1)
        # [CuPy] 使用 cp.sum
        accuracy = cp.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        self.loss(x, t, train_flg=True)
        dout = 1
        dout = self.loss_layer.backward(dout)
        layers_to_backprop = list(self.layers)
        layers_to_backprop.reverse()
        for layer in layers_to_backprop:
            dout = layer.backward(dout)

        grads = {
            "W1": self.layers[0].dW,
            "b1": self.layers[0].db,
            "gamma1": self.layers[1].dgamma,
            "beta1": self.layers[1].dbeta,
            "W2": self.layers[4].dW,
            "b2": self.layers[4].db,
            "gamma2": self.layers[5].dgamma,
            "beta2": self.layers[5].dbeta,
        }
        return grads


# --- 优化器 (已修改为 CuPy) ---
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
                # [CuPy] 使用 cp.zeros_like
                self.m[key] = cp.zeros_like(params[key])
                self.v[key] = cp.zeros_like(params[key])
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (
                grads[key] ** 2
            )
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            # [CuPy] 使用 cp.sqrt
            params[key] -= self.lr * m_hat / (cp.sqrt(v_hat) + self.epsilon)


# --- 主训练流程 (已修改为 CuPy) ---
if __name__ == "__main__":
    print("Loading MNIST dataset...")
    mnist = sklearn.datasets.fetch_openml(
        "mnist_784", version=1, as_frame=False, parser="auto"
    )
    data = mnist.data
    target = mnist.target
    print("Dataset loaded.")

    # [CuPy] 将数据从 NumPy 数组转移到 CuPy 数组 (CPU -> GPU)
    # 建议使用 float32 以获得更好的 GPU 性能
    print("Moving data to GPU...")
    x_train_cpu = data[:60000]
    t_train_cpu = target[:60000].astype(cp.int32)
    x_test_cpu = data[60000:]
    t_test_cpu = target[60000:].astype(cp.int32)

    x_train = cp.asarray(x_train_cpu, dtype=cp.float32)
    t_train_raw = cp.asarray(t_train_cpu)
    x_test = cp.asarray(x_test_cpu, dtype=cp.float32)
    t_test_raw = cp.asarray(t_test_cpu)

    # 在 GPU 上进行归一化和 one-hot 编码
    x_train /= 255.0
    x_test /= 255.0
    t_train = cp.eye(10, dtype=cp.float32)[t_train_raw]
    t_test = cp.eye(10, dtype=cp.float32)[t_test_raw]
    print("Data moved and preprocessed on GPU.")

    epoch_num = 20
    batch_size = 100
    hidden_size = 256

    network = TwoLayerNetWithLayers(
        input_size=784, hidden_size=hidden_size, output_size=10, dropout_ratio=0.5
    )
    optimizer = Adam(learning_rate=0.001)

    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size // batch_size, 1)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for epoch in range(epoch_num):
        # [CuPy] 在 GPU 上生成随机索引
        idx = cp.random.permutation(train_size)
        x_train_shuffled = x_train[idx]
        t_train_shuffled = t_train[idx]

        for i in tqdm(range(iter_per_epoch), desc=f"Epoch {epoch+1}/{epoch_num}"):
            batch_x = x_train_shuffled[i * batch_size : (i + 1) * batch_size]
            batch_t = t_train_shuffled[i * batch_size : (i + 1) * batch_size]

            grads = network.gradient(batch_x, batch_t)
            optimizer.update(network.params, grads)

            loss = network.loss(batch_x, batch_t)
            # [CuPy] 将 loss 值从 GPU 传回 CPU 以便存储在 Python 列表中
            train_loss_list.append(loss.item())

        # 计算准确率时，数据已在 GPU 上，计算会很快
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)

        # [CuPy] 将准确率值从 GPU 传回 CPU
        train_acc_list.append(train_acc.item())
        test_acc_list.append(test_acc.item())

        last_loss = train_loss_list[-1]
        print(
            f"Epoch {epoch + 1}/{epoch_num} done. Loss: {last_loss:.4f}, Train Acc: {train_acc.item():.4f}, Test Acc: {test_acc.item():.4f}"
        )

    # Matplotlib 需要 NumPy 数组，所以绘图前数据已在 CPU
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    plt.subplot(1, 2, 2)
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
        f"Final Train Accuracy: {train_acc_list[-1]:.4f}, Final Test Accuracy: {test_acc_list[-1]:.4f}"
    )
