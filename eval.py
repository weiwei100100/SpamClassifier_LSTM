import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import jieba
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 检查设备
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载词汇表
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)


# 定义中文分词器函数
def chinese_tokenizer(text):
    return list(jieba.cut(text))


# 定义自定义数据集类
class SpamDataset(Dataset):
    def __init__(self, data, vocab, tokenizer=None, max_len=100):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer if tokenizer else chinese_tokenizer
        self.max_len = max_len
        self.vocab = vocab

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


# 加载数据
data = pd.read_csv('processed_emails.csv')

# 划分训练集和测试集
from sklearn.model_selection import train_test_split

_, test_data = train_test_split(data, test_size=0.3, random_state=42)

# 创建测试数据集和数据加载器
test_dataset = SpamDataset(test_data, vocab=vocab)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 定义模型类
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


# 初始化模型
vocab_size = len(vocab)
embed_size = 128
hidden_size = 128
num_layers = 2

model = SpamClassifierLSTM(vocab_size, embed_size, hidden_size, num_layers).to(device)

# 加载模型参数
model.load_state_dict(torch.load('lstm_spam_classifier.pth', map_location=device))
model.eval()

# 模型评估
all_labels = []
all_predictions = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        probs = torch.sigmoid(outputs)
        predictions = (probs > 0.5).float().squeeze(1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

# 计算指标
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, zero_division=0)
recall = recall_score(all_labels, all_predictions, zero_division=0)
f1 = f1_score(all_labels, all_predictions, zero_division=0)

print(f"模型评估结果：")
print(f"准确率（Accuracy）：{accuracy:.4f}")
print(f"精确率（Precision）：{precision:.4f}")
print(f"召回率（Recall）：{recall:.4f}")
print(f"F1 值（F1 Score）：{f1:.4f}")