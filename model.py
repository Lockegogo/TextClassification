
import torch
from torch import nn
import torchkeras


class Net(nn.Module):
    def __init__(self, max_words=10000, embed_dim=3):
        super(Net, self).__init__()

        # 设置 padding_idx 参数后将在训练过程中将填充的 token 始终赋值为 0 向量
        self.embedding = nn.Embedding(
            num_embeddings=max_words, embedding_dim=embed_dim, padding_idx=1
        )
        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv_1", nn.Conv1d(in_channels=3, out_channels=16, kernel_size=5)
        )
        self.conv.add_module("pool_1", nn.MaxPool1d(kernel_size=2))
        self.conv.add_module("relu_1", nn.ReLU())
        self.conv.add_module(
            "conv_2", nn.Conv1d(in_channels=16, out_channels=128, kernel_size=2)
        )
        self.conv.add_module("pool_2", nn.MaxPool1d(kernel_size=2))
        self.conv.add_module("relu_2", nn.ReLU())

        self.dense = nn.Sequential()
        self.dense.add_module("flatten", nn.Flatten())
        self.dense.add_module("linear", nn.Linear(6144, 1))
        self.dense.add_module("sigmoid", nn.Sigmoid())

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.conv(x)
        y = self.dense(x)
        return y


class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

        self.correct = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.total = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, y_pred, y_true):
        y_pred = torch.where(
            y_pred > 0.5,
            torch.ones_like(y_pred, dtype=torch.float32),
            torch.zeros_like(y_pred, dtype=torch.float32),
        )
        # acc = torch.mean(1 - torch.abs(y_true - y_pred))
        m = (y_pred == y_true).sum().float()
        n = y_true.shape[0]
        self.correct += m
        self.total += n
        acc = m / n
        return acc

    def compute(self):
        return self.correct / self.total

    def reset(self):
        self.correct -= self.correct
        self.total -= self.total

