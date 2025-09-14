# /// script
# dependencies = ['torch']
# ///

import torch
import torch.nn as nn
import torch.optim as optim

# ==============================================================================
# 1. 数据准备 (Data Preparation)
# ==============================================================================

# 我们使用一个简单的英文句子作为我们的语料库
corpus = "hello world this is a simple rnn language model"
tokens = corpus.split(' ')

# 创建词汇表：将每个唯一的词映射到一个整数索引
# word_to_idx: {'hello': 0, 'world': 1, ...}
word_to_idx = {word: i for i, word in enumerate(sorted(list(set(tokens))))}
# idx_to_word: {0: 'hello', 1: 'world', ...}
idx_to_word = {i: word for word, i in word_to_idx.items()}

vocab_size = len(word_to_idx)
print(f"词汇表大小 (Vocabulary Size): {vocab_size}")
print(f"词汇表 (Vocabulary): {word_to_idx}")

# 创建训练数据
# 语言模型的任务是根据前面的词来预测下一个词。
# input: "hello world this is a simple rnn language"
# target: "world this is a simple rnn language model"
input_indices = [word_to_idx[word] for word in tokens[:-1]]
target_indices = [word_to_idx[word] for word in tokens[1:]]

# 将数据转换为PyTorch张量 (Tensors)
# .unsqueeze(0) 是为了增加一个 batch 维度，因为PyTorch模型通常期望输入是 (batch_size, sequence_length)
inputs = torch.LongTensor(input_indices).unsqueeze(0)
targets = torch.LongTensor(target_indices).unsqueeze(0)

print(f"\n输入序列 (Input Sequence): {inputs.shape}\n{inputs}")
print(f"目标序列 (Target Sequence): {targets.shape}\n{targets}")


# ==============================================================================
# 2. 模型定义 (Model Definition)
# ==============================================================================

class SimpleRNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        """
        初始化模型层
        :param vocab_size: 词汇表的大小
        :param embedding_dim: 词嵌入向量的维度
        :param hidden_dim: RNN隐藏层的维度
        """
        super().__init__()
        # 嵌入层：将词的索引转换为密集向量（embedding）
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # RNN层：处理序列数据，batch_first=True表示输入的第一个维度是batch_size
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

        # 全连接层（输出层）：将RNN的隐藏状态映射到整个词汇表的分数
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        """
        定义模型的前向传播
        :param x: 输入张量，形状为 (batch_size, sequence_length)
        :param hidden: RNN的初始隐藏状态
        :return: 输出logits和新的隐藏状态
        """
        # 1. 嵌入
        # 输入x: (batch_size, seq_len) -> embed_out: (batch_size, seq_len, embedding_dim)
        embed_out = self.embedding(x)

        # 2. RNN
        # rnn_out: (batch_size, seq_len, hidden_dim) - 每个时间步的隐藏状态
        # hidden: (1, batch_size, hidden_dim) - 最后一个时间步的隐藏状态
        rnn_out, hidden = self.rnn(embed_out, hidden)

        # 3. 全连接层
        # 将每个时间步的输出都映射到词汇表空间
        # rnn_out: (batch_size, seq_len, hidden_dim) -> logits: (batch_size, seq_len, vocab_size)
        logits = self.fc(rnn_out)

        return logits, hidden

# ==============================================================================
# 3. 训练过程 (Training Process)
# ==============================================================================

# 设置超参数
EMBEDDING_DIM = 10  # 词嵌入维度
HIDDEN_DIM = 32     # RNN隐藏层维度
LEARNING_RATE = 0.01
EPOCHS = 300

# 实例化模型
model = SimpleRNNLM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM)

# 定义损失函数和优化器
# CrossEntropyLoss 用于多分类问题，它内部包含了Softmax操作
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\n开始训练...")

# 训练循环
for epoch in range(EPOCHS):
    # 初始化隐藏状态 (对于每个新的epoch或序列)
    # 形状为 (num_layers, batch_size, hidden_dim)
    hidden = torch.zeros(1, 1, HIDDEN_DIM)

    # 清零梯度
    optimizer.zero_grad()

    # 前向传播
    outputs, hidden = model(inputs, hidden)

    # 计算损失
    # CrossEntropyLoss期望的输入形状是 (N, C) 和 (N)
    # outputs: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
    # targets: (batch_size, seq_len) -> (batch_size * seq_len)
    loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')

print("训练完成！")


# ==============================================================================
# 4. 结果测试 (Inference/Testing)
# ==============================================================================

print("\n--- 测试模型预测 ---")
# 使用训练好的模型进行预测
# 让我们看看给定 "hello" 之后，模型会预测什么
test_input_word = "hello"
test_input = torch.LongTensor([[word_to_idx[test_input_word]]])
hidden = torch.zeros(1, 1, HIDDEN_DIM)

with torch.no_grad(): # 在测试阶段不计算梯度
    outputs, _ = model(test_input, hidden)
    # 获取预测结果中分数最高的词的索引
    predicted_idx = outputs.argmax(dim=2).item()
    predicted_word = idx_to_word[predicted_idx]

print(f"输入: '{test_input_word}'")
print(f"模型预测的下一个词是: '{predicted_word}'")
print(f"实际的下一个词是: 'world'")
