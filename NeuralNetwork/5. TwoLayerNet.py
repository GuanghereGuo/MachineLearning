import numpy as np
import matplotlib.pyplot as plt
import sklearn
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

class TwoLayerNet:
    # one hot encoding

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.prams = {}
        self.prams['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.prams['b1'] = np.zeros(hidden_size)
        self.prams['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.prams['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, b1 = self.prams['W1'], self.prams['b1']
        W2, b2 = self.prams['W2'], self.prams['b2']

        a1 = np.dot(x, W1) + b1
        # z1 = sigmoid(a1)
        z1 = relu(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    # mini-batch
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    # not mini-batch
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
            'W1': numerical_gradient(loss_W, self.prams['W1']),
            'b1': numerical_gradient(loss_W, self.prams['b1']),
            'W2': numerical_gradient(loss_W, self.prams['W2']),
            'b2': numerical_gradient(loss_W, self.prams['b2'])
        }

        return grads

    def gradient(self, x, t):
        W1, b1 = self.prams['W1'], self.prams['b1']
        W2, b2 = self.prams['W2'], self.prams['b2']

        batch_num = x.shape[0]

        # 前向传播
        a1 = np.dot(x, W1) + b1
        #z1 = sigmoid(a1)
        z1 = relu(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # 反向传播
        dy = (y - t) / batch_num
        grads = {'W2': np.dot(z1.T, dy), 'b2': np.sum(dy, axis=0)}

        da1 = np.dot(dy, W2.T)
        #dz1 = da1 * (z1 * (1 - z1))
        dz1 = da1 * (z1 > 0)  # ReLU的梯度
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads


if __name__ == '__main__':

    # 获取训练集和测试集
    mnist = sklearn.datasets.fetch_openml('mnist_784', version=1, as_frame=False)
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

    # 设置超参数
    epoch_num = 20
    batch_size = 100
    learning_rate = 0.1

    train_size = x_train.shape[0]
    iter_per_epoch = train_size // batch_size
    total_iters = iter_per_epoch * epoch_num

    # 初始化神经网络
    net = TwoLayerNet(input_size=28*28, hidden_size=120, output_size=10)

    # 训练神经网络
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for epoch in range(epoch_num):
        print(f'Epoch {epoch + 1}/{epoch_num}')
        # 打乱训练数据
        indices = np.random.permutation(train_size)
        x_train = x_train[indices]
        t_train = t_train[indices]
        # 创建批次
        batches = np.array_split(indices, iter_per_epoch)
        assert len(batches) == iter_per_epoch, "Batch size does not match the number of iterations per epoch."
        for i in tqdm(range(iter_per_epoch)):
            batch_indices = batches[i % iter_per_epoch]
            x_batch = x_train[batch_indices]
            t_batch = t_train[batch_indices]

            # 计算梯度
            grads = net.gradient(x_batch, t_batch)

            # 更新参数
            for key in net.prams.keys():
                net.prams[key] -= learning_rate * grads[key]

            # 记录损失和准确率
            loss = net.loss(x_batch, t_batch)
            train_loss_list.append(loss)

        train_acc = net.accuracy(x_train, t_train)
        test_acc = net.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"Epoch {epoch+1} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    # 绘制损失和准确率曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='Train Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(test_acc_list, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.show()

    print(f"Final Train Accuracy: {train_acc_list[-1]:.4f}")
    print(f"Final Test Accuracy: {test_acc_list[-1]:.4f}")
    print("Training complete.")
