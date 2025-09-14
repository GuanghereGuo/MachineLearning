# /// script
# dependencies = ['torch']
# ///

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# 1. 数据准备 (Data Preparation)
# ==============================================================================

# 使用一个包含多个不同长度句子的语料库
corpus = [
    "hello world",
    "this is a simple rnn",
    "pytorch is fun",
    "a simple language model",
]

# -- 1.1 创建词汇表 --
# 将所有句子合并，并添加特殊标记 <PAD>
tokens = " ".join(corpus).split(" ")
vocab = sorted(list(set(tokens)))
vocab.append("<PAD>")  # 添加填充标记

print(tokens)
print(vocab)

word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}
vocab_size = len(word_to_idx)
pad_idx = word_to_idx["<PAD>"]

print(f"词汇表大小 (Vocabulary Size): {vocab_size}")
print(f"填充符索引 (Padding Index): {pad_idx}")
print(f"词汇表 (Vocabulary): {word_to_idx}")

# -- 1.2 创建输入/目标对 --
sequences = []
for sentence in corpus:
    indices = [word_to_idx[word] for word in sentence.split(" ")]
    # input: [idx_1, idx_2, ..., idx_n-1]
    # target: [idx_2, idx_3, ..., idx_n]
    sequences.append((torch.tensor(indices[:-1]), torch.tensor(indices[1:])))


# -- 1.3 自定义数据集 --
# PyTorch的DataLoader需要一个Dataset对象
class LanguageModelDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


# -- 1.4 定义 Collate Function --
# 这是处理 mini-batch 的核心！
# DataLoader 会将从 Dataset 中取出的样本列表传递给这个函数
def collate_fn(batch):
    """
    处理一个批次的数据：
    1. 按长度降序排序（pack_padded_sequence的要求）
    2. 填充序列，使它们在批次内长度一致
    3. 返回填充后的输入/目标张量以及原始长度
    """
    # batch 是一个元组列表: [(input1, target1), (input2, target2), ...]
    # 按输入序列的长度进行降序排序
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    inputs, targets = zip(*batch)

    # 获取每个输入序列的原始长度
    lengths = [len(seq) for seq in inputs]

    # 对输入和目标进行填充
    # nn.utils.rnn.pad_sequence 会自动处理填充
    padded_inputs = nn.utils.rnn.pad_sequence(
        inputs, batch_first=True, padding_value=pad_idx
    )
    padded_targets = nn.utils.rnn.pad_sequence(
        targets, batch_first=True, padding_value=pad_idx
    )

    return padded_inputs, padded_targets, torch.tensor(lengths)


# ==============================================================================
# 2. 模型定义 (Model Definition)
# ==============================================================================


class RNNLMWithBatch(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pad_idx):
        super(RNNLMWithBatch, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, lengths):
        """
        前向传播，处理填充后的批次数据
        :param x: 输入张量, shape: (batch_size, seq_len)
        :param lengths: 包含批次中每个序列原始长度的张量
        :return: logits
        """
        # 1. 嵌入
        # x: (batch_size, seq_len) -> embed_out: (batch_size, seq_len, embedding_dim)
        embed_out = self.embedding(x)

        # 2. 打包序列
        # 告诉RNN忽略填充部分
        packed_embed = nn.utils.rnn.pack_padded_sequence(
            embed_out, lengths, batch_first=True
        )

        # 3. RNN
        # RNN只处理打包后的有效数据
        packed_rnn_out, _ = self.rnn(packed_embed)

        # 4. 解包序列
        # 将输出恢复为填充后的形状，以便后续层处理
        # rnn_out shape: (batch_size, seq_len, hidden_dim)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(packed_rnn_out, batch_first=True)

        # 5. 全连接层
        # logits shape: (batch_size, seq_len, vocab_size)
        logits = self.fc(rnn_out)

        return logits


# ==============================================================================
# 3. 训练过程 (Training Process)
# ==============================================================================

# -- 超参数 --
EMBEDDING_DIM = 16
HIDDEN_DIM = 32
LEARNING_RATE = 0.01
EPOCHS = 500
BATCH_SIZE = 2

# -- 实例化数据加载器 --
dataset = LanguageModelDataset(sequences)
# DataLoader 会使用 collate_fn 来组合单个样本，形成一个 mini-batch
train_loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)

# -- 实例化模型、损失函数、优化器 --
model = RNNLMWithBatch(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, pad_idx)

# 使用 ignore_index 参数，损失函数会自动忽略所有目标为 pad_idx 的位置！
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\n开始训练...")

# -- 训练循环 --
for epoch in range(EPOCHS):
    for batch_inputs, batch_targets, batch_lengths in train_loader:
        # batch_inputs shape: (batch_size, max_len_in_batch)
        # batch_targets shape: (batch_size, max_len_in_batch)
        # batch_lengths shape: (batch_size,)

        optimizer.zero_grad()

        # 前向传播
        outputs = model(batch_inputs, batch_lengths)

        # 计算损失
        # CrossEntropyLoss 期望 (N, C) 和 (N)
        # outputs: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
        # batch_targets: (batch_size, seq_len) -> (batch_size * seq_len)
        loss = criterion(outputs.view(-1, vocab_size), batch_targets.view(-1))

        loss.backward()
        optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

print("训练完成！")

# ==============================================================================
# 4. 结果测试 (Inference/Testing)
# ==============================================================================
print("\n--- 测试模型预测 ---")
test_sentence = "this is a"
test_indices = [word_to_idx[w] for w in test_sentence.split(" ")]
test_input = torch.LongTensor([test_indices])
# 测试时，长度就是序列本身的长度，因为只有一个样本，不需要打包
test_lengths = torch.LongTensor([len(test_indices)])

with torch.no_grad():
    outputs = model(test_input, test_lengths)
    # 我们只关心最后一个时间步的输出，来预测下一个词
    last_word_logits = outputs[0, -1, :]
    predicted_idx = last_word_logits.argmax().item()
    predicted_word = idx_to_word[predicted_idx]

print(f"输入: '{test_sentence}'")
print(f"模型预测的下一个词是: '{predicted_word}'")
print(f"实际可能的下一个词是: 'simple'")
