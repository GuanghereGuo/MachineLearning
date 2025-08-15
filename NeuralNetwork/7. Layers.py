import numpy as np
import matplotlib.pyplot as plt
import sklearn
from abc import ABC, abstractmethod

from tqdm import tqdm


def softmax(x):
    if x.ndim == 2:  # 如果是二维数组
        x = x - np.max(x, axis=1, keepdims=True)  # 每行减去最大值
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
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x) # 梯度数组与 x 形状相同
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index # 获取当前元素的多元索引 (例如: (0, 1), (2, 0, 3) 等)
        tmp_val = x[idx]     # 保存当前元素的值
        # 计算 f(x+h)
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        # 计算 f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        # 计算偏导数并存储到梯度数组的对应位置
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val # 还原值
        it.iternext() # 移动到下一个元素
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
        out = np.maximum(0, x)
        return out

    def backward(self, dout):
        dx = np.zeros_like(self.x)
        dx[self.x > 0] = dout[self.x > 0]
        return dx

class Affine(Layer):
    def __init__(self, W, b):
        super().__init__()
        self.x = None
        self.W = W # 不可以copy！！
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

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weight_init_std = weight_init_std
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = [
            Affine(self.params['W1'], self.params['b1']),
            ReLu(),
            Affine(self.params['W2'], self.params['b2'])
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

    def numerical_gradient(self, x, t):
        def loss_W(_):
            return self.loss(x, t)

        grads = {
            'W1': numerical_gradient(loss_W, self.params['W1']),
            'b1': numerical_gradient(loss_W, self.params['b1']),
            'W2': numerical_gradient(loss_W, self.params['W2']),
            'b2': numerical_gradient(loss_W, self.params['b2'])
        }

        return grads

    def gradient(self, x, t):
        for layer in self.layers:
            x = layer.forward(x)

        dout = self.loss_layer.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        grads = {
            'W1': self.layers[0].dW,
            'b1': self.layers[0].db,
            'W2': self.layers[2].dW,
            'b2': self.layers[2].db
        }

        return grads

if __name__ == '__main__':
    mnist = sklearn.datasets.fetch_openml("mnist_784", version=1, as_frame=False)
    data = mnist.data
    target = mnist.target

    # 数据预处理

    x_train = data[:60000]  # 取前60000个样本作为训练集
    t_train = target[:60000].astype(np.int32)

    x_test = data[60000:]
    t_test = target[60000:].astype(np.int32)

    x_test = x_test / 255.0  # 归一化
    x_test = x_test.reshape(-1, 28 * 28)
    t_test = np.eye(10)[t_test]  # one-hot encoding

    x_train = x_train / 255.0  # 归一化
    x_train = x_train.reshape(-1, 28 * 28)  # 展平为一维数组
    t_train = np.eye(10)[t_train]  # one-hot encoding

    # 创建神经网络
    network = TwoLayerNetWithLayers(input_size=28*28, hidden_size=200, output_size=10)

    epoch_num = 20
    batch_size = 100
    learning_rate = 0.2

    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size // batch_size, 1)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for epoch in range(epoch_num):
        idx = np.random.permutation(train_size)
        x_train = x_train[idx]
        t_train = t_train[idx]

        for i in tqdm(range(iter_per_epoch)):
            batch_x = x_train[i * batch_size:(i + 1) * batch_size]
            batch_t = t_train[i * batch_size:(i + 1) * batch_size]

            # 计算损失
            loss = network.loss(batch_x, batch_t)
            train_loss_list.append(loss)

            # 计算梯度
            grads = network.gradient(batch_x, batch_t)

            # 更新参数
            for key in network.params.keys():
                network.params[key] -= learning_rate * grads[key]

        # 计算训练集和测试集的准确率
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(f'Epoch {epoch + 1}/{epoch_num}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    # 绘制损失和准确率曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='Train Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(test_acc_list, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()
    print("Training complete.")
    print(f"Final Train Accuracy: {train_acc_list[-1]:.4f}, Final Test Accuracy: {test_acc_list[-1]:.4f}")