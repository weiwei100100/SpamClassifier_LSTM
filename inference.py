import torch
import torch.nn as nn
import jieba
import pickle
import pandas as pd

# 设备配置
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")


# 定义中文分词器函数
def chinese_tokenizer(text):
    return list(jieba.cut(text))


# 定义模型类，与训练时保持一致
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


# 加载词汇表
def load_vocab():
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    return vocab


# 加载词汇表和模型
vocab = load_vocab()
tokenizer = chinese_tokenizer
vocab_size = len(vocab)


# 加载模型
def load_model(model_path, vocab_size, embed_size=128, hidden_size=128, num_layers=2):
    model = SpamClassifierLSTM(vocab_size, embed_size, hidden_size, num_layers)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# 定义文本预处理函数
def preprocess_text(text, vocab, tokenizer, max_len=100):
    tokens = tokenizer(text)
    token_indices = [vocab[token] for token in tokens[:max_len]]
    if len(token_indices) < max_len:
        token_indices += [vocab["<pad>"]] * (max_len - len(token_indices))
    else:
        token_indices = token_indices[:max_len]
    input_tensor = torch.tensor(token_indices, dtype=torch.long).unsqueeze(0)  # 添加批次维度
    return input_tensor


# 定义推理函数
def predict(text, model, vocab, tokenizer, max_len=100):
    input_tensor = preprocess_text(text, vocab, tokenizer, max_len).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output)
        # 获取预测类别
        prediction = (prob > 0.5).float()
        # 根据预测类别调整置信度
        if prediction.item() == 1.0:
            confidence = prob.item()  # 预测为垃圾邮件，置信度为 prob_spam
        else:
            confidence = 1 - prob.item()  # 预测为正常邮件，置信度为 1 - prob_spam
    return prediction.item(), confidence


# 主程序
if __name__ == "__main__":
    # 加载模型
    model = load_model('lstm_spam_classifier.pth', vocab_size)

    while True:
        print("\n请选择操作：")
        print("1. 手动输入文本进行预测")
        print("2. 使用随机生成的样例数据进行预测")
        print("3. 结束预测并退出程序")
        choice = input("请输入您的选择（1/2/3）：")

        if choice == '1':
            # 手动输入
            user_input = input("\n请输入一段文本，让模型判断是否为垃圾邮件：\n")
            prediction, confidence = predict(user_input, model, vocab, tokenizer)
            label = "垃圾邮件" if prediction == 1.0 else "正常邮件"
            print(f"\n模型预测结果：{label}（置信度：{confidence:.4f}）")
        elif choice == '2':
            # 随机生成
            sample_data = pd.read_csv('processed_emails.csv')
            sample_text = sample_data.sample(1)['text'].values[0]
            sample_text_t = sample_text.replace(" ", "")
            print("\n随机生成的样例文本：")
            print(sample_text_t)
            prediction, confidence = predict(sample_text, model, vocab, tokenizer)
            label = "垃圾邮件" if prediction == 1.0 else "正常邮件"
            print(f"\n模型预测结果：{label}（置信度：{confidence:.4f}）")
        elif choice == '3':
            print("\n程序已退出。")
            break
        else:
            print("\n无效的选择，请输入 1、2 或 3。")
