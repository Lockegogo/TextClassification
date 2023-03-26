import argparse
import logging
import time
import os
import string, re
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch import nn
import torchtext
import torchkeras
from torchkeras import summary
import torchmetrics
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from torchkeras.kerascallbacks import TensorBoardCallback

# if use multi gpu
from accelerate import notebook_launcher
from accelerate import Accelerator
from accelerate.utils import write_basic_config
from accelerate.utils import set_seed

write_basic_config()

from model import Net, Accuracy
from dataloader import MyDataLoader, read_csv, preprocess_text
from utils import plot_metric

r"""
This file shows the training process of the text classification model.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Classification Training")
    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        default=False,
        help="use multi gpu (default: False)",
    )
    # num_processes: 使用多少个 gpu 进行训练
    parser.add_argument("--num_processes", type=int, default=4, help="num processes")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch size (default: 64)"
    )
    parser.add_argument(
        "--max_len", type=int, default=500, help="max length of text (default: 500)"
    )
    # max_words：仅考虑训练集中出现频率最高的 max_words 个词
    parser.add_argument("--max_words", type=int, default=10000)
    parser.add_argument(
        "--epochs", type=int, default=20, help="training epochs (default: 20)"
    )
    parser.add_argument(
        "--embed_dim", type=int, default=32, help="embedding dimension (default: 32)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.02, help="learning rate (default: 0.02)"
    )
    parser.add_argument("--patience", type=int, default=3, help="patience (default: 3)")
    # log_interval: 每隔多少个 batch 打印一次训练信息
    parser.add_argument(
        "--log_interval", type=int, default=10, help="log interval (default: 10)"
    )
    parser.add_argument(
        "--logging-level", default="INFO", help="logging level (default=INFO)"
    )
    # 是否使用 tensorboard
    parser.add_argument(
        "--use_tensorboard",
        action="store_true",
        default=False,
        help="use tensorboard (default: False)",
    )
    args = parser.parse_args()
    num_processes = args.num_processes
    batch_size = args.batch_size
    max_len = args.max_len
    max_words = args.max_words
    epochs = args.epochs
    embed_dim = args.embed_dim
    lr = args.lr
    patience = args.patience
    log_interval = args.log_interval
    logging_level = args.logging_level
    multi_gpu = args.multi_gpu
    use_tensorboard = args.use_tensorboard

    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, args.logging_level),
    )
    logger = logging.getLogger("LK")

    torch.random.seed()
    root_path = './data/imdb'
    read_csv(root_path, 'IMDB Dataset.csv')
    train_iter, test_iter, TEXT, ds_train, ds_test = preprocess_text(
        root_path, max_words=max_words, max_len=max_len, batch_size=batch_size
    )

    # 查看数据集信息
    logger.info(f"ds_train[0].text: {ds_train[0].text}")
    logger.info(f"ds_train[0].label: {ds_train[0].label}")
    # 查看词表信息
    logger.info(f"TEXT.vocab.stoi['<unk>']: {TEXT.vocab.stoi['<unk>']}")
    logger.info(f"TEXT.vocab.stoi['<pad>']: {TEXT.vocab.stoi['<pad>']}")
    logger.info(f"TEXT.vocab.itos[0]: {TEXT.vocab.itos[0]}")
    logger.info(f"TEXT.vocab.freqs['good']: {TEXT.vocab.freqs['good']}")

    # 查看数据迭代器信息
    for batch in train_iter:
        features = batch.text
        labels = batch.label
        logger.info("DataLoader:")
        logger.info(f"features.shape: {features.shape}")
        logger.info(f"labels.shape: {labels.shape}")
        break

    dl_train = MyDataLoader(train_iter)
    dl_test = MyDataLoader(test_iter)

    # 查看数据加载器信息
    for features, labels in dl_train:
        logger.info("Revised DataLoader:")
        logger.info(f"features.shape: {features.shape}")
        logger.info(f"labels.shape: {labels.shape}")
        break

    net = Net(max_words, embed_dim)

    logger.info("Model Architecture:")
    logger.info(net)

    model = torchkeras.KerasModel(
        net,
        loss_fn=nn.BCELoss(),
        optimizer=torch.optim.Adagrad(net.parameters(), lr=lr),
        ## 可以直接使用内置的
        # metrics_dict = {'acc':torchmetrics.Accuracy(task='binary')}
        metrics_dict={"acc": Accuracy()},
    )

    summary(model, input_data=features)

    ## if use multi gpu
    if multi_gpu:
        args = dict(
            train_data=dl_train,
            val_data=dl_test,
            epochs=epochs,
            ckpt_path='checkpoints/checkpoint.pt',
            patience=patience,
            monitor='val_acc',
            mode='max',
            mixed_precision='no',
        ).values()

        # num_processes: 使用多少个进程进行训练
        notebook_launcher(model.fit, args, num_processes=num_processes)

        # 评估模型
        model.net.load_state_dict(torch.load('checkpoints/checkpoint.pt'))
        logger.info(model.evaluate(dl_test))
    else:
        if use_tensorboard:
            # callbacks: 传入一个列表，列表中可以包含多个回调函数
            # log_weight: 是否记录权重
            # log_weight_freq: 记录权重的频率, 即每隔多少个 epoch 记录一次权重
            dfhistory = model.fit(
                train_data=dl_train,
                val_data=dl_test,
                epochs=epochs,
                patience=patience,
                ckpt_path='checkpoints/checkpoint.pt',
                monitor="val_acc",
                mode="max",
                callbacks=[
                    TensorBoardCallback(
                        save_dir='tensorboards',
                        model_name='text_cls',
                        log_weight=True,
                        log_weight_freq=5,
                    )
                ],
            )
            # 在命令行输入：
            # tensorboard --logdir="./tensorboards" --bind_all --port=6006 --purge_orphaned_data=true
            # 在浏览器中打开 http://localhost:6006/ 查看
            # 如果是服务器：http://10.192.9.235:20038/，具体地址需要查看服务器的说明
        else:
            dfhistory = model.fit(
                train_data=dl_train,
                val_data=dl_test,
                epochs=epochs,
                patience=patience,
                ckpt_path='checkpoints/checkpoint.pt',
                monitor="val_acc",
                mode="max",
                # plot=True,
            )

        # 评估模型
        plot_metric(dfhistory, "loss")
        plot_metric(dfhistory, "acc")
        logger.info(model.evaluate(dl_test))

        # 使用模型
        net = model.net
        net.eval()

        model_device = next(net.parameters()).device
        for features, labels in dl_test:
            y_prob = net(features.to(model_device))
            y_pred = torch.where(
                y_prob > 0.5,
                torch.ones_like(y_prob, dtype=torch.float32),
                torch.zeros_like(y_prob, dtype=torch.float32),
            )
            logger.info("Use model for prediction (1 batch):")
            logger.info(f"y_pred: {y_pred}")
            logger.info(f"y_true: {labels}")
            break

        # 保存模型
        net_clone = Net()

        model_clone = torchkeras.KerasModel(
            net_clone,
            loss_fn=nn.BCELoss(),
            optimizer=torch.optim.Adagrad(net.parameters(), lr=lr),
            metrics_dict={"accuracy": Accuracy()},
        )

        model_clone.net.load_state_dict(torch.load("checkpoints/checkpoint.pt"))

        # 评估模型
        model_clone.evaluate(dl_test)

