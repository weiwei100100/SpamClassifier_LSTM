import os
import pandas as pd
from bs4 import BeautifulSoup
import jieba
import string

# 文件夹路径
spam_folder = './data/spam'
ham_folder = './data/ham'


# 文本清洗与预处理函数
def preprocess_text(text):
    # 使用 BeautifulSoup 去除 HTML 标签
    soup = BeautifulSoup(text, 'html.parser')
    clean_text = soup.get_text()

    # 使用 jieba 进行中文分词
    tokens = jieba.lcut(clean_text)

    # 去除标点符号和空白字符
    tokens = [word for word in tokens if word.strip() and word not in string.punctuation]

    return ' '.join(tokens)


# 从文件夹中读取邮件并返回 DataFrame
def load_emails_from_folder(folder_path, label):
    email_data = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            email_content = file.read()
            processed_content = preprocess_text(email_content)
            email_data.append((processed_content, label))
    return email_data


# 加载垃圾邮件和正常邮件
spam_emails = load_emails_from_folder(spam_folder, label=1)
ham_emails = load_emails_from_folder(ham_folder, label=0)

# 合并为一个 DataFrame
all_emails = pd.DataFrame(spam_emails + ham_emails, columns=['text', 'label'])

# 保存数据到CSV文件
output_file = 'processed_emails.csv'
all_emails.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"数据已成功保存到 {output_file}")