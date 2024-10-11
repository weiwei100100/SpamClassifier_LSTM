import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import jieba
import pickle

# 检查设备
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载数据
data = pd.read_csv('processed_emails.csv')

# 划分训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)


# 定义中文分词器函数
def chinese_tokenizer(text):
    return list(jieba.cut(text))


# 定义自定义数据集类
class SpamDataset(Dataset):
    def __init__(self, data, vocab=None, tokenizer=None, max_len=100):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer if tokenizer else chinese_tokenizer
        self.max_len = max_len
        self.vocab = vocab if vocab else self.build_vocab(data['text'])

    def build_vocab(self, texts):
        vocab = build_vocab_from_iterator(self.tokenize_texts(texts), specials=["<unk>", "<pad>"])
        vocab.set_default_index(vocab["<unk>"])
        return vocab

    def tokenize_texts(self, texts):
        for text in texts:
            yield self.tokenizer(text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']
        tokens = self.tokenizer(text)
        token_indices = [self.vocab[token] for token in tokens[:self.max_len]]
        if len(token_indices) < self.max_len:
            token_indices += [self.vocab["<pad>"]] * (self.max_len - len(token_indices))
        else:
            token_indices = token_indices[:self.max_len]
        return torch.tensor(token_indices, dtype=torch.long), torch.tensor(label, dtype=torch.float32)


# 初始化数据集和词汇表
train_dataset = SpamDataset(train_data)
test_dataset = SpamDataset(test_data, vocab=train_dataset.vocab)

# 保存词汇表到文件
with open('vocab.pkl', 'wb') as f:
    pickle.dump(train_dataset.vocab, f)

print("词汇表已保存到 vocab.pkl 文件中。")

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义 LSTM 模型
class SpamClassifierLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(SpamClassifierLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = h_n[-1, :, :]
        out = self.fc(out)
        return out


# 初始化模型参数
vocab_size = len(train_dataset.vocab)
embed_size = 128
hidden_size = 128
num_layers = 2
model = SpamClassifierLSTM(vocab_size, embed_size, hidden_size, num_layers).to(device)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# 在测试集上评估模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        predicted = (torch.sigmoid(outputs) > 0.5).float().squeeze(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'测试集准确率: {accuracy:.4f}')

# 保存模型
torch.save(model.state_dict(), 'lstm_spam_classifier.pth')
print("LSTM 模型已成功保存。")
