import tkinter as tk
from tkinter import messagebox, scrolledtext
import torch
import torch.nn as nn
import jieba
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset

# 设备配置
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载词汇表
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# 定义中文分词器函数
def chinese_tokenizer(text):
    return list(jieba.cut(text))

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

# 加载模型
vocab_size = len(vocab)
embed_size = 128
hidden_size = 128
num_layers = 2

model = SpamClassifierLSTM(vocab_size, embed_size, hidden_size, num_layers).to(device)
model.load_state_dict(torch.load('lstm_spam_classifier.pth', map_location=device))
model.eval()

# 定义文本预处理函数
def preprocess_text(text, vocab, tokenizer, max_len=100):
    tokens = tokenizer(text)
    token_indices = []
    for token in tokens[:max_len]:
        try:
            index = vocab[token]
        except KeyError:
            index = vocab['<unk>']
        token_indices.append(index)
    if len(token_indices) < max_len:
        token_indices += [vocab["<pad>"]] * (max_len - len(token_indices))
    else:
        token_indices = token_indices[:max_len]
    input_tensor = torch.tensor(token_indices, dtype=torch.long).unsqueeze(0)  # 添加批次维度
    return input_tensor

# 定义推理函数
def predict(text):
    input_tensor = preprocess_text(text, vocab, chinese_tokenizer).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output)
        prediction = (prob > 0.5).float()
        if prediction.item() == 1.0:
            confidence = prob.item()
        else:
            confidence = 1 - prob.item()
    label = "垃圾邮件" if prediction.item() == 1.0 else "正常邮件"
    return label, confidence

# 自定义数据集类用于评估
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
        token_indices = []
        for token in tokens[:self.max_len]:
            try:
                index = self.vocab[token]
            except KeyError:
                index = self.vocab['<unk>']
            token_indices.append(index)
        if len(token_indices) < self.max_len:
            token_indices += [self.vocab["<pad>"]] * (self.max_len - len(token_indices))
        else:
            token_indices = token_indices[:self.max_len]
        return torch.tensor(token_indices, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

# 定义模型评估函数
def evaluate_model():
    # 加载数据
    data = pd.read_csv('processed_emails.csv')
    # 划分训练集和测试集
    from sklearn.model_selection import train_test_split
    _, test_data = train_test_split(data, test_size=0.3, random_state=42)
    # 创建测试数据集和数据加载器
    test_dataset = SpamDataset(test_data, vocab=vocab)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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

    result = (f"模型评估结果：\n"
              f"准确率（Accuracy）：{accuracy:.4f}\n"
              f"精确率（Precision）：{precision:.4f}\n"
              f"召回率（Recall）：{recall:.4f}\n"
              f"F1 值（F1 Score）：{f1:.4f}")
    return result

# 构建 GUI
def main():
    window = tk.Tk()
    window.title("垃圾邮件检测")
    window.geometry("600x800")
    window.resizable(False, False)

    # 设置全局字体
    default_font = ("Helvetica", 14)

    # 主框架
    main_frame = tk.Frame(window, padx=10, pady=10)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # 输入与预测结果区域
    input_frame = tk.LabelFrame(main_frame, text="输入与预测结果", padx=10, pady=10, font=("Helvetica", 14))
    input_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    input_label = tk.Label(input_frame, text="请输入一段文本，让模型判断是否为垃圾邮件：", font=default_font)
    input_label.pack(anchor='w')

    input_text = tk.Text(input_frame, height=5, font=default_font)
    input_text.pack(fill=tk.BOTH, expand=True)

    # 预测按钮
    def on_predict():
        text = input_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("输入为空", "请输入文本后再进行预测。")
            return
        label, confidence = predict(text)
        result_text.config(state='normal')
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, f"模型预测结果：{label}（置信度：{confidence:.4f}）")
        result_text.config(state='disabled')

    predict_button = tk.Button(input_frame, text="预测", command=on_predict, width=10, font=default_font)
    predict_button.pack(pady=5)

    result_label = tk.Label(input_frame, text="预测结果：", font=default_font)
    result_label.pack(anchor='w')

    result_text = tk.Text(input_frame, height=2, font=default_font, state='disabled')
    result_text.pack(fill=tk.BOTH, expand=True)

    # 随机样本预测区域
    random_frame = tk.LabelFrame(main_frame, text="随机样本预测", padx=10, pady=10, font=("Helvetica", 14))
    random_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    random_text_label = tk.Label(random_frame, text="随机生成的样例文本：", font=default_font)
    random_text_label.pack(anchor='w')

    random_text = scrolledtext.ScrolledText(random_frame, height=5, font=default_font)
    random_text.pack(fill=tk.BOTH, expand=True)

    # 随机预测按钮
    def on_random_predict():
        # 加载数据
        data = pd.read_csv('processed_emails.csv')
        # 随机选择一条样本
        sample = data.sample(1).iloc[0]
        sample_text = sample['text']
        sample_text_t = sample_text.replace(" ", "")
        sample_label = sample['label']
        # 在界面上显示样本文本
        random_text.delete("1.0", tk.END)
        random_text.insert(tk.END, sample_text_t)
        # 进行预测
        label, confidence = predict(sample_text)
        result_random_text.config(state='normal')
        result_random_text.delete("1.0", tk.END)
        result_random_text.insert(tk.END, f"模型预测结果：{label}（置信度：{confidence:.4f}）\n")
        # 显示真实标签
        true_label = "垃圾邮件" if sample_label == 1 else "正常邮件"
        result_random_text.insert(tk.END, f"真实标签：{true_label}")
        result_random_text.config(state='disabled')

    random_predict_button = tk.Button(random_frame, text="随机预测", command=on_random_predict, width=10, font=default_font)
    random_predict_button.pack(pady=5)

    result_random_label = tk.Label(random_frame, text="随机预测结果：", font=default_font)
    result_random_label.pack(anchor='w')

    result_random_text = tk.Text(random_frame, height=2, font=default_font, state='disabled')
    result_random_text.pack(fill=tk.BOTH, expand=True)

    # 模型评估区域
    eval_frame = tk.LabelFrame(main_frame, text="模型评估", padx=10, pady=10, font=("Helvetica", 14))
    eval_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    eval_text = scrolledtext.ScrolledText(eval_frame, height=3, font=default_font)
    eval_text.pack(fill=tk.BOTH, expand=True)

    # 评估按钮
    def on_evaluate():
        eval_result = evaluate_model()
        eval_text.delete("1.0", tk.END)
        eval_text.insert(tk.END, eval_result)

    evaluate_button = tk.Button(eval_frame, text="评估模型", command=on_evaluate, width=10, font=default_font)
    evaluate_button.pack(pady=5)

    window.mainloop()


if __name__ == "__main__":
    main()