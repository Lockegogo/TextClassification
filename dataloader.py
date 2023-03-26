import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import re
import string
import torchtext
import torch

# 读取 csv 文件
def read_csv(root_path, file_name):
    file_path = os.path.join(root_path, file_name)
    df = pd.read_csv(file_path)
    df['text'] = df['review'].apply(lambda x: x.replace('<br />', ' '))
    df['label'] = np.where(df['sentiment'] == 'positive', 1, 0)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv(
        os.path.join(root_path, 'train.tsv'),
        sep='\t',
        index=False,
        header=False,
        columns=['label', 'text'],
    )
    test_df.to_csv(
        os.path.join(root_path, 'test.tsv'),
        sep='\t',
        index=False,
        header=False,
        columns=['label', 'text'],
    )


# 文本数据预处理
def preprocess_text(root_path, max_words=10000, max_len=500, batch_size=64):
    train_df = pd.read_csv(
        os.path.join(root_path, 'train.tsv'),
        sep='\t',
        header=None,
        names=['label', 'text'],
    )
    test_df = pd.read_csv(
        os.path.join(root_path, 'test.tsv'),
        sep='\t',
        header=None,
        names=['label', 'text'],
    )
    tokenizer = lambda x: re.sub('[%s]' % string.punctuation, "", x).split(" ")

    # 过滤掉低频词
    def filter_low_freq_words(arr, vocab):
        arr = [[x if x < max_words else 0 for x in example] for example in arr]
        return arr

    TEXT = torchtext.data.Field(
        sequential=True,  # 是否是序列数据
        tokenize=tokenizer,  # 分词方法
        lower=True,  # 是否将英文字母转为小写
        fix_length=max_len,  # 每个样本的固定长度
        postprocessing=filter_low_freq_words,  # 处理低频词的方法
    )

    LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

    # 构建表格型 dataset，读取数据集并将数据按照指定字段进行处理
    ds_train, ds_test = torchtext.data.TabularDataset.splits(
        path=root_path,  # 数据集的路径
        train='train.tsv',  # 训练集文件名
        test='test.tsv',  # 测试集文件名
        format='tsv',  # 数据格式
        fields=[('label', LABEL), ('text', TEXT)],  # 指定字段
        skip_header=False,  # 是否跳过文件头
    )

    # 构建词汇表
    TEXT.build_vocab(ds_train, min_freq=5)

    # 将数据集转换为迭代器
    train_iter, test_iter = torchtext.data.Iterator.splits(
        (ds_train, ds_test),  # 数据集
        sort_within_batch=True,  # 是否在每个批次内按文本长度排序
        sort_key=lambda x: len(x.text),  # 排序方式
        batch_sizes=(batch_size, batch_size),  # 批次大小
    )

    return train_iter, test_iter, TEXT, ds_train, ds_test


# 将数据管道组织成 torch.utils.data.DataLoader 相似的 features, label 输出形式
class MyDataLoader():
    def __init__(self, data_iter):
        self.data_iter = data_iter
        self.length = len(data_iter)

    def __len__(self):
        return self.length

    def __iter__(self):
        # 注意：此处调整 features为 batch first，并调整 label 的 shape 和 dtype
        for batch in self.data_iter:
            yield (
                torch.transpose(batch.text, 0, 1),
                torch.unsqueeze(batch.label.float(), dim=1),
            )