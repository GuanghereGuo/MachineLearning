import torch
import torch.nn as nn
import os
import math
from collections import Counter
from tqdm import tqdm

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()

    def add_word(self, word):
        self.counter[word] += 1

    def build_vocab(self):
        # Add <unk> token first
        self.idx2word.append('<unk>')
        self.word2idx['<unk>'] = 0

        sorted_words = sorted(self.counter.keys(), key=lambda word: self.counter[word], reverse=True)
        for word in sorted_words:
            if word not in self.word2idx:
                self.idx2word.append(word)
                self.word2idx[word] = len(self.idx2word) - 1

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, path='./ptb_data'):
        self.dictionary = Dictionary()

        self.build_vocab(os.path.join(path, 'ptb.train.txt'))

        self.train = self.tokenize(os.path.join(path, 'ptb.train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'ptb.valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'ptb.test.txt'))

    def build_vocab(self, path):
        assert os.path.exists(path)
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)
        self.dictionary.build_vocab()

    def tokenize(self, path):
        assert os.path.exists(path)
        with open(path, 'r', encoding="utf8") as f:
            ids = []
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids.append(self.dictionary.word2idx.get(word, self.dictionary.word2idx['<unk>']))
            return torch.LongTensor(ids)


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.5, tie_weights=True):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, vocab_size)

        if tie_weights:
            if hidden_size != embed_size:
                raise ValueError('When using the tied flag, hidden_size must be equal to embed_size')
            self.decoder.weight = self.embedding.weight

        self.init_weights()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, hidden):
        emb = self.embedding(x)
        emb = self.drop(emb)
        output, hidden = self.lstm(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded.view(-1, decoded.size(2)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.hidden_size),
                weight.new_zeros(self.num_layers, bsz, self.hidden_size))


embed_size = 256
hidden_size = 256
num_layers = 2
num_epochs = 40
batch_size = 20
bptt_len = 35
learning_rate = 20.0
dropout = 0.5
clip_grad = 0.25

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

corpus = Corpus()
vocab_size = len(corpus.dictionary)

def batchify(data, bsz):
    num_batches = data.size(0) // bsz
    data = data.narrow(0, 0, num_batches * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

train_data = batchify(corpus.train, batch_size)
val_data = batchify(corpus.valid, batch_size)
test_data = batchify(corpus.test, batch_size)


model = RNNModel(vocab_size, embed_size, hidden_size, num_layers, dropout, tie_weights=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i):
    seq_len = min(bptt_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def evaluate(data_source, desc="Evaluating"):
    model.eval()
    total_loss = 0.
    eval_batch_size = data_source.size(1)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        # 使用tqdm进行评估
        data_iterator = range(0, data_source.size(0) - 1, bptt_len)
        for i in tqdm(data_iterator, desc=desc, leave=False, unit="batch"):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            total_loss += len(data) * criterion(output, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)

def train():
    model.train()
    total_loss = 0.
    hidden = model.init_hidden(batch_size)

    # 使用tqdm创建进度条
    num_batches = (train_data.size(0) - 1) // bptt_len
    data_iterator = range(0, train_data.size(0) - 1, bptt_len)

    progress_bar = tqdm(enumerate(data_iterator), total=num_batches, desc=f"Epoch {epoch:2d}", unit="batch")

    for batch, i in progress_bar:
        data, targets = get_batch(train_data, i)
        hidden = repackage_hidden(hidden)

        output, hidden = model(data, hidden)
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item()

        # 使用tqdm的set_postfix方法动态更新信息
        if batch % 10 == 0: # 更新频率可以调整
            current_lr = optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix(
                loss=f'{loss.item():.2f}',
                ppl=f'{math.exp(loss.item()):8.2f}',
                lr=f'{current_lr:.4f}'
            )

best_val_loss = float("inf")

try:
    for epoch in range(1, num_epochs + 1):
        train()
        val_loss = evaluate(val_data, desc=f"Validating Epoch {epoch:2d}")
        print() # 在tqdm进度条后换行
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | valid loss {val_loss:5.2f} | '
              f'valid ppl {math.exp(val_loss):8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            print("Validation loss improved. Saving model...")
            with open('model.pt', 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            print("Validation loss did not improve. Decaying learning rate.")
            scheduler.step()

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# 加载最佳模型并进行最终测试
print("\nLoading best model for final test...")
with open('model.pt', 'rb') as f:
    # ***** FIX: Set weights_only=False *****
    # This is required in PyTorch 2.6+ when loading a full model object (not just a state_dict)
    # It tells PyTorch to trust the source and allow unpickling the model's class structure.
    model = torch.load(f, weights_only=False)

model.to(device)

test_loss = evaluate(test_data, desc="Final Test")
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | test ppl {math.exp(test_loss):8.2f}')
print('=' * 89)
