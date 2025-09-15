# /// script
# dependencies = ['torch', 'tqdm']
# ///

import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import os
import collections
import sys
from tqdm import tqdm # <-- 1. 引入 tqdm

# ==============================================================================
# 1. 设定超参数和设备
# ==============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 20
EVAL_BATCH_SIZE = 10
EMBED_SIZE = 200
HIDDEN_SIZE = 200
NUM_LAYERS = 2
BPTT_LEN = 35
DROPOUT = 0.5
LEARNING_RATE = 0.001 # <-- 2. 为 Adam 优化器设置一个合适的学习率
EPOCHS = 40
CLIP = 0.25

# ==============================================================================
# 2. 数据加载与预处理 (从本地文件)
# ==============================================================================
print("正在从本地加载数据...")

def check_data_files(data_path='ptb_data'):
    files = ['ptb.train.txt', 'ptb.valid.txt', 'ptb.test.txt']
    file_paths = []

    if not os.path.exists(data_path):
        print(f"错误: 数据文件夹 '{data_path}' 不存在。")
        sys.exit(1)

    for filename in files:
        filepath = os.path.join(data_path, filename)
        if not os.path.exists(filepath):
            print(f"错误: 数据文件 '{filepath}' 不存在。")
            sys.exit(1)
        file_paths.append(filepath)

    print("数据文件检查通过。")
    return file_paths

class PTBCorpus:
    def __init__(self, file_paths):
        self.train_path, self.valid_path, self.test_path = file_paths
        self.build_vocab()

        self.train = self.file_to_ids(self.train_path)
        self.valid = self.file_to_ids(self.valid_path)
        self.test = self.file_to_ids(self.test_path)

    def build_vocab(self):
        with open(self.train_path, 'r', encoding='utf-8') as f:
            tokens = f.read().replace('\n', ' <eos> ').split()
        counter = collections.Counter(tokens)
        sorted_words = sorted(counter, key=counter.get, reverse=True)
        self.word_to_id = {word: i for i, word in enumerate(sorted_words)}
        self.id_to_word = {i: word for i, word in enumerate(sorted_words)}
        self.vocab_size = len(self.word_to_id)

    def file_to_ids(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            tokens = f.read().replace('\n', ' <eos> ').split()
        unk_id = self.word_to_id.get('<unk>')
        ids = [self.word_to_id.get(word, unk_id) for word in tokens]
        return torch.tensor(ids, dtype=torch.long)

file_paths = check_data_files()
corpus = PTBCorpus(file_paths)
vocab_size = corpus.vocab_size
print(f"词汇表大小: {vocab_size}")

def batchify(data, bsz):
    num_batches = data.size(0) // bsz
    data = data.narrow(0, 0, num_batches * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

train_data = batchify(corpus.train, BATCH_SIZE)
val_data = batchify(corpus.valid, EVAL_BATCH_SIZE)
test_data = batchify(corpus.test, EVAL_BATCH_SIZE)
print(f"训练数据形状 (seq_len, batch_size): {train_data.shape}")

# ==============================================================================
# 3. 模型定义 (Stateful LSTM)
# ==============================================================================
class RNNLM(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNLM, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.lstm = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, hidden):
        emb = self.drop(self.encoder(src))
        output, hidden = self.lstm(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))

# ==============================================================================
# 4. 训练与评估逻辑 - 修改了 train() 函数
# ==============================================================================
model = RNNLM(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
criterion = nn.CrossEntropyLoss()

def get_batch(source, i):
    seq_len = min(BPTT_LEN, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def train():
    model.train()
    hidden = model.init_hidden(BATCH_SIZE)

    # <-- 3. 使用 tqdm 包裹迭代器
    pbar = tqdm(range(0, train_data.size(0) - 1, BPTT_LEN),
                desc=f"Epoch {epoch:2d}/{EPOCHS:2d}",
                ncols=100) # ncols 控制进度条宽度

    for i in pbar:
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, vocab_size), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()

        # <-- 4. 更新tqdm进度条的后缀信息
        pbar.set_postfix(loss=loss.item(), ppl=math.exp(loss.item()))

def evaluate(data_source):
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(data_source.size(1))
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, BPTT_LEN):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, vocab_size), targets)
            total_loss += len(data) * loss.item()
    return total_loss / (len(data_source) - 1)

# ==============================================================================
# 5. 主训练循环 - 修改了优化器
# ==============================================================================
best_val_loss = float('inf')

# <-- 5. 将优化器从 SGD 更换为 Adam
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 学习率调度器对于 Adam 依然有用，但其行为可能与 SGD 不同
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train() # train 函数现在会显示进度条
    val_loss = evaluate(val_data)
    print() # 在tqdm进度条结束后换行
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {time.time() - epoch_start_time:5.2f}s | '
          f'valid loss {val_loss:5.2f} | valid ppl {math.exp(val_loss):8.2f}')
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open('best_model.pt', 'wb') as f:
            torch.save(model.state_dict(), f)
    else:
        # 如果验证集上的性能没有提升，可以考虑调整学习率
        scheduler.step()

# ==============================================================================
# 6. 在测试集上最终评估
# ==============================================================================
model = RNNLM(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
with open('best_model.pt', 'rb') as f:
    model.load_state_dict(torch.load(f))

test_loss = evaluate(test_data)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | test ppl {math.exp(test_loss):8.2f}')
print('=' * 89)
