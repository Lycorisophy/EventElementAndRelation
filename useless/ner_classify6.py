import os
import re
import torch
import math
import logging
import argparse
import threading
import torch.optim
import numpy as np
import torch.nn as nn
from tqdm import trange
from albert_master.my_logger import Logger
from albert_master.optimization import Ranger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from albert_master.tokenization_albert import FullTokenizer, convert_single_example
from albert_master.modeling_albert_bright import AlbertConfig, AlbertModel
from attention import SuperDualDecoder as SDD
from attention import DualDecoder as DD
from attention import SuperEncoder as SE
from attention import BiDecoder as BD


def read_lines(file_name):
    lines = open(file_name, encoding='utf-8').read().strip().split('\n')
    return [one_line for one_line in lines]


def text2tensor(tokenizer, l_text, max_len):
    output_ids, output_mask, segment_ids = convert_single_example(tokenizer, l_text, max_len)
    tokens_tensor = torch.tensor([output_ids])
    masks_tensor = torch.tensor([output_mask])
    segments_tensors = torch.tensor([segment_ids])
    return tokens_tensor, masks_tensor, segments_tensors


def sent2vec(l_model, tokenizer, l_text, max_len):
    tokens_tensor, masks_tensor, segments_tensors = text2tensor(tokenizer, l_text, max_len)
    embedding, _ = l_model(tokens_tensor, segments_tensors, masks_tensor, False)
    return embedding


def label_from_output(output):
    _, top_i = output.data.topk(1)
    return top_i[0]


def text2ids(tokenizer, l_text, max_len):
    tokens_tensor, masks_tensor, segments_tensors = text2tensor(tokenizer, l_text, max_len)
    return tokens_tensor.numpy(), masks_tensor.numpy(), segments_tensors.numpy()


def accuracy(preds, labels, seq_len):
    count, right = 0, 0
    for pred, label in zip(preds, labels):
        for i in range(seq_len):
            if label[i] != len(args.tag_to_ix)-1 and label[i] != len(args.tag_to_ix)-2 \
                    and label[i] != len(args.tag_to_ix)-3:
                count += 1
                _, p = pred[i].topk(1)
                if int(label[i]) == p[0].item():
                    right += 1
    return right/count


def accuracy2(preds, labels, seq_len):
    count, right = 0.1, 0.0
    for pred, label in zip(preds, labels):
        for i in range(seq_len):
            if label[i] != len(args.tag_to_ix)-1 and label[i] != len(args.tag_to_ix)-2 \
                    and label[i] != len(args.tag_to_ix)-3 and label[i] != len(args.tag_to_ix)-4:
                count += 1
                _, p = pred[i].topk(1)
                if int(label[i]) == p[0].item():
                    right += 1
    acc = right/count
    if acc > 0.99:
        return 0.99
    elif acc < 0.2:
        return 0.2
    else:
        return acc


def adjust_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.95


def bio2label(t_label, max_len, args):
    while len(t_label) > max_len-2:
        t_label.pop()
    t_label.append('[SEP]')
    while len(t_label) < max_len:
        t_label.append('[MASK]')
    return [args.tag_to_ix[t] for t in t_label]


def to_scalar(var):
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class MyLoss1(nn.Module):
    def __init__(self):
        super(MyLoss1, self).__init__()

    def forward(self, x, y):
        return -torch.sum((y*torch.log(x + 1e-10)))


class MyLoss2(nn.Module):
    def __init__(self):
        super(MyLoss2, self).__init__()
        self.tag_size = len(args.tag_to_ix)

    def forward(self, x):
        return -torch.sum(1/self.tag_size*torch.log(x + 1e-10))


class LstmNer1(nn.Module):
    def __init__(self, config):
        super(LstmNer1, self).__init__()
        self.tag_to_ix = args.tag_to_ix
        self.tagset_size = len(args.tag_to_ix)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, num_layers=2, bidirectional=True)
        self.dense = nn.Linear(config.hidden_size*2, self.tagset_size)
        self.soft = nn.Softmax(dim=-1)
        self.loss = MyLoss1()

    def _one_hot(self, x, tags):
        ys = torch.zeros_like(x)
        for y, tag in zip(ys, tags):
            for i in range(args.max_text_len):
                y[i][tag[i]] = 1
        return ys

    def _hard_hot(self, x, tags):
        ys = torch.ones_like(x)
        for y, tag in zip(ys, tags):
            for i in range(args.max_text_len):
                y[i][tag[i]] = 100
        return ys

    def get_pred(self, x):
        x, (_, _) = self.lstm(x)
        x = self.dense(x)
        return x

    def forward(self, x, tags):
        x, (_, _) = self.lstm(x)
        x = self.dense(x)
        x = self.soft(x)
        y = self._hard_hot(x, tags)
        acc1 = accuracy(x, tags.detach().cpu().numpy(), args.max_text_len)
        acc2 = accuracy2(x, tags.detach().cpu().numpy(), args.max_text_len)
        return self.loss(x, y, 1 - acc2), acc1


class MlaNer1(nn.Module):
    def __init__(self, config):
        super(MlaNer1, self).__init__()
        self.tagset_size = len(args.tag_to_ix)
        self.encoder = BD(config.hidden_size,
                           args.max_text_len,
                           config.num_hidden_layers2,
                           config.num_attention_heads,
                           1)
        self.dense = nn.Linear(config.hidden_size*2, self.tagset_size)
        self.soft = nn.Softmax(dim=-1)
        self.loss1 = MyLoss1()
        self.loss2 = MyLoss2()

    def _one_hot(self, x, tags):
        ys = torch.zeros_like(x)
        for y, tag in zip(ys, tags):
            for i in range(args.max_text_len):
                y[i][tag[i]] = 1
        return ys

    def _hard_hot(self, x, tags):
        ys = torch.ones_like(x)
        for y, tag in zip(ys, tags):
            for i in range(args.max_text_len):
                y[i][tag[i]] = 100
        return ys

    def _dynamic_target(self, x, tags, t):
        ys = torch.ones_like(x)
        for y, tag in zip(ys, tags):
            for i in range(args.max_text_len):
                tmp = tag[i]
                y[i][tmp] = args.tag_to_score[tmp.item()]**t
        return ys

    def _mask_hot(self, x, tags, masks):
        ys = torch.zeros_like(x)
        for y, tag, mask in zip(ys, tags, masks):
            for i in range(args.max_text_len):
                if mask[i] == 1:
                    y[i][tag[i]] = 1
        return ys

    def get_pred(self, x, attention_mask=None):
        x = self.encoder(x, attention_mask)
        x = self.dense(x)
        return x

    def forward(self, x, tags, attention_mask=None, t=1):
        x = self.encoder(x, attention_mask)
        x = self.dense(x)
        if attention_mask is not None:
            extended_attention_mask = torch.ones_like(x)
            extended_attention_mask = extended_attention_mask.permute(0, 2, 1)*attention_mask
            extended_attention_mask = extended_attention_mask.permute(0, 2, 1)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            x += extended_attention_mask
        x = self.soft(x)
        acc1 = accuracy(x, tags.detach().cpu().numpy(), args.max_text_len)
        # acc2 = accuracy2(x, tags.detach().cpu().numpy(), args.max_text_len)
        y = self._dynamic_target(x, tags, t)
        y = self.soft(y)
        loss1 = self.loss1(x, y)
        # loss2 = self.loss2(x)
        # loss = (1-acc2)*loss1+acc2*loss2
        return loss1, acc1


class MlaNer2(nn.Module):
    def __init__(self, config):
        super(MlaNer2, self).__init__()
        self.tag_to_ix = args.tag_to_ix
        self.tagset_size = len(args.tag_to_ix)
        self.dense = nn.Linear(config.hidden_size, self.tagset_size)
        self.soft = nn.Softmax(dim=-1)
        self.loss = MyLoss1()

    def _one_hot(self, x, tags):
        ys = torch.zeros_like(x)
        for y, tag in zip(ys, tags):
            for i in range(args.max_text_len):
                y[i][tag[i]] = 1
        return ys

    def _hard_hot(self, x, tags):
        ys = torch.ones_like(x)
        for y, tag in zip(ys, tags):
            for i in range(args.max_text_len):
                y[i][tag[i]] = 100
        return ys

    def get_pred(self, x):
        x = self.dense(x)
        return x

    def forward(self, x, tags):
        x = self.dense(x)
        x = self.soft(x)
        y = self._hard_hot(x, tags)
        acc1 = accuracy(x, tags.detach().cpu().numpy(), args.max_text_len)
        acc2 = accuracy2(x, tags.detach().cpu().numpy(), args.max_text_len)
        return self.loss(x, y, 1 - acc2), acc1


class MlaNer3(nn.Module):
    def __init__(self, config):
        super(MlaNer3, self).__init__()
        self.tag_to_ix = args.tag_to_ix
        self.tagset_size = len(args.tag_to_ix)
        self.encoder = SE(config.hidden_size,
                          config.num_hidden_layers2,
                          config.intermediate_size,
                          config.num_attention_heads)
        self.dense = nn.Linear(config.hidden_size, self.tagset_size)
        self.soft = nn.Softmax(dim=-1)
        self.loss = MyLoss1()

    def _one_hot(self, x, tags):
        ys = torch.zeros_like(x)
        for y, tag in zip(ys, tags):
            for i in range(args.max_text_len):
                y[i][tag[i]] = 1
        return ys

    def _hard_hot(self, x, tags):
        ys = torch.ones_like(x)
        for y, tag in zip(ys, tags):
            for i in range(args.max_text_len):
                y[i][tag[i]] = 100
        return ys

    def get_pred(self, x, attention_mask=None):
        x = self.encoder(x, attention_mask)
        x = self.dense(x)
        return x

    def forward(self, x, tags, attention_mask=None):
        x = self.encoder(x, attention_mask)
        x = self.dense(x)
        x = self.soft(x)
        y = self._hard_hot(x, tags)
        acc1 = accuracy(x, tags.detach().cpu().numpy(), args.max_text_len)
        acc2 = accuracy2(x, tags.detach().cpu().numpy(), args.max_text_len)
        return self.loss(x, y, acc2), acc1


def mla_train1(args, logger, train_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AlbertConfig.from_pretrained(args.config_mla1)
    encoder = AlbertModel(config=config)
    model = MlaNer1(config=config)
    encoder.from_pretrained('zh_large_bright/', config=config)
    encoder.to(device)
    model.to(device)
    param_optimizer1 = list(encoder.named_parameters())
    param_optimizer2 = list(model.named_parameters())
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in param_optimizer1 if any(nd in n for nd in ['embeddings'])],
         'weight_decay_rate': args.weight_decay,
         'lr': args.embeddings_lr},
        {'params': [p for n, p in param_optimizer1 if not any(nd in n for nd in ['embeddings'])],
         'lr': args.encoder_lr},
    ]
    optimizer_grouped_parameters2 = [
        {'params': [p for n, p in param_optimizer2 if any(nd in n for nd in ['encoder'])],
         'lr': args.encoder_lr},
        {'params': [p for n, p in param_optimizer2 if not any(nd in n for nd in ['encoder'])],
         'lr': args.learning_rate},
    ]
    optimizer1 = Ranger(optimizer_grouped_parameters1)
    optimizer2 = Ranger(optimizer_grouped_parameters2)
    epochs = args.train_epochs
    bio_records = []
    train_loss_set = []
    acc_records = []

    tokenizer = FullTokenizer(vocab_file=args.vocab_dir)
    input_ids = []
    mask_ids = []
    labels = []
    Text = re.compile('.+\t')
    Label = re.compile('\t.+')
    t_label = []
    text = ""
    for line in read_lines('data/test.txt'):
        if line != "":
            tmp = re.sub(u'\t', '', Text.findall(line)[0])
            text += tmp
            t_label.append(re.sub(u'\t', '', Label.findall(line)[0]))
        else:
            tmp1, tmp2, _ = text2ids(tokenizer, text, args.max_text_len)
            input_ids.append(tmp1)
            mask_ids.append(tmp2)
            labels.append(t_label)
            text = ""
            t_label = []
    test_inputs = torch.Tensor(input_ids).to(device)
    test_masks = torch.Tensor(mask_ids).to(device)

    for epoch in trange(epochs, desc='Epochs'):
        encoder.train()
        model.train()
        tr_loss = 0
        eval_loss, eval_accuracy = 0, 0
        nb_tr_steps = 0
        nb_eval_steps = 0
        tmp_loss = []
        t = 1-(epoch/epochs)/2
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            b_input_ids = b_input_ids.squeeze(1)
            b_input_mask = b_input_mask.squeeze(1)
            encoding = encoder(b_input_ids.long(), b_input_mask)
            loss, tmp_eval_accuracy = model(encoding[0], b_labels, b_input_mask, t)
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            torch.cuda.empty_cache()
            tr_loss += loss.item()
            nb_tr_steps += 1
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
            tmp_loss.append(loss.item())
            if step % 200 == 100:
                print('mla1训练中，训练损失:{:.2f},准确率为：{:.2f}%' .format(loss.item(), 100 * tmp_eval_accuracy))
        # adjust_learning_rate(optimizer1)
        # adjust_learning_rate(optimizer2)
        try:
            train_loss_set.append(tr_loss / nb_tr_steps)
            logger.info('mla1训练损失:{:.2f},准确率为：{:.2f}%'
                        .format(tr_loss / nb_tr_steps, 100 * eval_accuracy / nb_eval_steps))
            acc_records.append(eval_accuracy / nb_eval_steps)
            bio_records.append(np.mean(train_loss_set))
        except ZeroDivisionError:
            logger.info("错误！请降低batch大小")


        encoder.eval()
        model.eval()
        tag_seqs = []
        for test_input, test_mask in zip(test_inputs, test_masks):
            test_input = test_input.squeeze(1)
            test_mask = test_mask.squeeze(1)
            with torch.no_grad():
                encoding = encoder(test_input.long(), test_mask)
                tag_seq = model.get_pred(encoding[0], test_mask)
            tag_seqs.append(tag_seq)
        matrix = np.zeros((len(args.tag_to_idx), len(args.tag_to_idx)))
        for tag_seq, label in zip(tag_seqs, labels):
            tag_seq = tag_seq.squeeze(0)
            for i, la in enumerate(label):
                if i < args.max_text_len - 1:
                    _, p = tag_seq[i + 1].topk(1)
                    pre = args.ix_to_idx[p[0].item()]
                    true = args.tag_to_idx[args.tag_to_tag[la]]
                    matrix[true][pre] += 1
        size = len(args.tag_to_idx)
        P, R = [], []
        print("P")
        for i in range(size):
            t = np.sum(matrix[:, i])
            if t != 0:
                P.append(matrix[i][i] / t)
                print(matrix[i][i] / t)
            else:
                P.append(0)
                print(0)
        print("R")
        for i in range(size):
            t = np.sum(matrix[i])
            if t != 0:
                R.append(matrix[i][i] / t)
                print(matrix[i][i] / t)
            else:
                R.append(0)
                print(0)
        print("F1")
        for i in range(size):
            if P[i] + R[i] != 0:
                print(2 * P[i] * R[i] / (P[i] + R[i]))
            else:
                print(0)
        # encoder_to_save = encoder.module if hasattr(encoder, 'module') else encoder
        # torch.save(encoder_to_save.state_dict(),
        #            os.path.join(os.path.join(args.mla_dir1, 'encoder/'), "pytorch_model201244d.bin"))
        # model_to_save = model.module if hasattr(model, 'module') else model
        # torch.save(model_to_save.state_dict(), os.path.join(args.mla_dir1, "pytorch_model201244d.bin"))
    return [encoder, model]


def mla_train2(args, logger, train_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AlbertConfig.from_pretrained(args.config_mla2)
    encoder = AlbertModel(config=config)
    model = MlaNer2(config=config)
    try:
        encoder.from_pretrained(os.path.join(args.mla_dir2, 'encoder/'), config=config)
    except:
        print("PretrainedMlaEncoder2NotFound")
        encoder.from_pretrained('zh_large_bright/', config=config)
    try:
        output_model_file = os.path.join(args.mla_dir2, "pytorch_model.bin")
        model_state_dict = torch.load(output_model_file)
        model.load_state_dict(model_state_dict)
    except:
        print("PretrainedMlaModel2NotFound")
    encoder.to(device)
    model.to(device)
    param_optimizer1 = list(encoder.named_parameters())
    param_optimizer2 = list(model.named_parameters())
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in param_optimizer1 if any(nd in n for nd in ['embeddings'])],
         'weight_decay_rate': args.weight_decay,
         'lr': args.embeddings_lr},
        {'params': [p for n, p in param_optimizer1 if not any(nd in n for nd in ['embeddings'])],
         'lr': args.encoder_lr},
    ]
    optimizer_grouped_parameters2 = [
        {'params': [p for n, p in param_optimizer2 if any(nd in n for nd in ['encoder'])],
         'lr': args.encoder_lr},
        {'params': [p for n, p in param_optimizer2 if not any(nd in n for nd in ['encoder'])],
         'lr': args.learning_rate},
    ]
    optimizer1 = Ranger(optimizer_grouped_parameters1)
    optimizer2 = Ranger(optimizer_grouped_parameters2)
    epochs = args.train_epochs
    bio_records = []
    train_loss_set = []
    acc_records = []
    encoder.train()
    model.train()
    for _ in trange(epochs, desc='Epochs'):
        tr_loss = 0
        eval_loss, eval_accuracy = 0, 0
        nb_tr_steps = 0
        nb_eval_steps = 0
        tmp_loss = []
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            b_input_ids = b_input_ids.squeeze(1)
            b_input_mask = b_input_mask.squeeze(1)
            encoding = encoder(b_input_ids.long(), b_input_mask)
            loss, tmp_eval_accuracy = model(encoding[0], b_labels)
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            torch.cuda.empty_cache()
            tr_loss += loss.item()
            nb_tr_steps += 1
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
            tmp_loss.append(loss.item())
            if step % 200 == 150:
                print('mla2训练中，训练损失:{:.2f},准确率为：{:.2f}%' .format(loss.item(), 100 * tmp_eval_accuracy))
        adjust_learning_rate(optimizer1)
        adjust_learning_rate(optimizer2)
        encoder_to_save = encoder.module if hasattr(encoder, 'module') else encoder
        torch.save(encoder_to_save.state_dict(),
                   os.path.join(os.path.join(args.mla_dir2, 'encoder/'), "pytorch_model.bin"))
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), os.path.join(args.mla_dir2, "pytorch_model.bin"))
        try:
            train_loss_set.append(tr_loss / nb_tr_steps)
            logger.info('mla2训练损失:{:.2f},准确率为：{:.2f}%'
                        .format(tr_loss / nb_tr_steps, 100 * eval_accuracy / nb_eval_steps))
            acc_records.append(eval_accuracy / nb_eval_steps)
            bio_records.append(np.mean(train_loss_set))
        except ZeroDivisionError:
            logger.info("错误！请降低batch大小")
    return [encoder, model]


def mla_train3(args, logger, train_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AlbertConfig.from_pretrained(args.config_mla3)
    encoder = AlbertModel(config=config)
    model = MlaNer3(config=config)
    try:
        encoder.from_pretrained(os.path.join(args.mla_dir3, 'encoder/'), config=config)
    except:
        print("PretrainedMlaEncoder3NotFound")
        encoder.from_pretrained('zh_large_bright/', config=config)
    try:
        output_model_file = os.path.join(args.mla_dir3, "pytorch_model.bin")
        model_state_dict = torch.load(output_model_file)
        model.load_state_dict(model_state_dict)
    except:
        print("PretrainedMlaModel3NotFound")
    encoder.to(device)
    model.to(device)
    param_optimizer1 = list(encoder.named_parameters())
    param_optimizer2 = list(model.named_parameters())
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in param_optimizer1 if any(nd in n for nd in ['embeddings'])],
         'weight_decay_rate': args.weight_decay,
         'lr': args.embeddings_lr},
        {'params': [p for n, p in param_optimizer1 if not any(nd in n for nd in ['embeddings'])],
         'lr': args.encoder_lr},
    ]
    optimizer_grouped_parameters2 = [
        {'params': [p for n, p in param_optimizer2],
         'lr': args.learning_rate},
    ]
    optimizer1 = Ranger(optimizer_grouped_parameters1)
    optimizer2 = Ranger(optimizer_grouped_parameters2)
    epochs = args.train_epochs
    bio_records = []
    train_loss_set = []
    acc_records = []
    encoder.train()
    model.train()
    for _ in trange(epochs, desc='Epochs'):
        tr_loss = 0
        eval_loss, eval_accuracy = 0, 0
        nb_tr_steps = 0
        nb_eval_steps = 0
        tmp_loss = []
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            b_input_ids = b_input_ids.squeeze(1)
            b_input_mask = b_input_mask.squeeze(1)
            encoding = encoder(b_input_ids.long(), b_input_mask)
            loss, tmp_eval_accuracy = model(encoding[0], b_labels, b_input_mask)
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            torch.cuda.empty_cache()
            tr_loss += loss.item()
            nb_tr_steps += 1
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
            tmp_loss.append(loss.item())
            if step % 200 == 150:
                print('mla3训练中，训练损失:{:.2f},准确率为：{:.2f}%' .format(loss.item(), 100 * tmp_eval_accuracy))
        adjust_learning_rate(optimizer1)
        adjust_learning_rate(optimizer2)
        try:
            train_loss_set.append(tr_loss / nb_tr_steps)
            logger.info('mla3训练损失:{:.2f},准确率为：{:.2f}%'
                        .format(tr_loss / nb_tr_steps, 100 * eval_accuracy / nb_eval_steps))
            acc_records.append(eval_accuracy / nb_eval_steps)
            bio_records.append(np.mean(train_loss_set))
        except ZeroDivisionError:
            logger.info("错误！请降低batch大小")
    encoder_to_save = encoder.module if hasattr(encoder, 'module') else encoder
    torch.save(encoder_to_save.state_dict(),
               os.path.join(os.path.join(args.mla_dir3, 'encoder/'), "pytorch_model.bin"))
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), os.path.join(args.mla_dir3, "pytorch_model.bin"))
    return [encoder, model]


def lstm_train1(args, logger, train_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AlbertConfig.from_pretrained(args.config_lstm1)
    encoder = AlbertModel(config=config)
    model = LstmNer1(config=config)
    try:
        encoder.from_pretrained(os.path.join(args.lstm_dir1, 'encoder/'), config=config)
    except:
        print("PretrainedLstmEncoder1NotFound")
        encoder.from_pretrained('zh_large_bright/', config=config)
    try:
        output_model_file = os.path.join(args.lstm_dir1, "pytorch_model.bin")
        model_state_dict = torch.load(output_model_file)
        model.load_state_dict(model_state_dict)
    except:
        print("PretrainedLstmModel1NotFound")
    encoder.to(device)
    model.to(device)
    param_optimizer1 = list(encoder.named_parameters())
    param_optimizer2 = list(model.named_parameters())
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in param_optimizer1 if any(nd in n for nd in ['embeddings'])],
         'weight_decay_rate': args.weight_decay,
         'lr': args.embeddings_lr},
        {'params': [p for n, p in param_optimizer1 if not any(nd in n for nd in ['embeddings'])],
         'lr': args.encoder_lr},
    ]
    optimizer_grouped_parameters2 = [
        {'params': [p for n, p in param_optimizer2],
         'lr': args.learning_rate},
    ]
    optimizer1 = Ranger(optimizer_grouped_parameters1)
    optimizer2 = Ranger(optimizer_grouped_parameters2)
    epochs = args.train_epochs
    bio_records = []
    train_loss_set = []
    acc_records = []
    encoder.train()
    model.train()
    for _ in trange(epochs, desc='Epochs'):
        tr_loss = 0
        eval_loss, eval_accuracy = 0, 0
        nb_tr_steps = 0
        nb_eval_steps = 0
        tmp_loss = []
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            b_input_ids = b_input_ids.squeeze(1)
            b_input_mask = b_input_mask.squeeze(1)
            encoding = encoder(b_input_ids.long(), b_input_mask)
            loss, tmp_eval_accuracy = model(encoding[0], b_labels)
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            torch.cuda.empty_cache()
            tr_loss += loss.item()
            nb_tr_steps += 1
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
            tmp_loss.append(loss.item())
            if step % 200 == 150:
                print('lstm1训练中，训练损失:{:.2f},准确率为：{:.2f}%' .format(loss.item(), 100 * tmp_eval_accuracy))
        adjust_learning_rate(optimizer1)
        adjust_learning_rate(optimizer2)
        encoder_to_save = encoder.module if hasattr(encoder, 'module') else encoder
        torch.save(encoder_to_save.state_dict(),
                   os.path.join(os.path.join(args.lstm_dir1, 'encoder/'), "pytorch_model.bin"))
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), os.path.join(args.lstm_dir1, "pytorch_model.bin"))
        try:
            train_loss_set.append(tr_loss / nb_tr_steps)
            logger.info('lstm1训练损失:{:.2f},准确率为：{:.2f}%'
                        .format(tr_loss / nb_tr_steps, 100 * eval_accuracy / nb_eval_steps))
            acc_records.append(eval_accuracy / nb_eval_steps)
            bio_records.append(np.mean(train_loss_set))
        except ZeroDivisionError:
            logger.info("错误！请降低batch大小")
    return [encoder, model]


def get_dataloader(filenames):
    # 读取训练数据
    albert_tokenizer = FullTokenizer(vocab_file=args.vocab_dir)
    input_ids = []
    mask_ids = []
    labels = []
    cnt = 0
    Text = re.compile('.+\t')
    Label = re.compile('\t.+')
    t_label = ['[CLS]']
    text = ""
    for line in read_lines(filenames):
        if line != "":
            text += re.sub(u'\t', '', Text.findall(line)[0])
            t_label.append(re.sub(u'\t', '', Label.findall(line)[0]))
        if line == "":
            tmp1, tmp2, _ = text2ids(albert_tokenizer, text, args.max_text_len)
            label = bio2label(t_label, args.max_text_len, args)
            input_ids.append(tmp1)
            mask_ids.append(tmp2)
            labels.append(label)
            cnt += 1
            text = ""
            t_label = ['[CLS]']
    train_input, validation_input, train_mask, validation_mask, train_labels, validation_labels = \
        train_test_split(input_ids, mask_ids, labels, random_state=args.seed, test_size=0)

    # 将训练集、验证集转化成tensor
    train_inputs = torch.Tensor(train_input)
    train_masks = torch.Tensor(train_mask)
    train_labels = torch.LongTensor(train_labels)

    # 生成dataloader
    batch_size = args.train_batch_size
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=batch_size)
    return train_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 可调参数
    parser.add_argument("--train_epochs",
                        default=20,   # 默认5
                        type=int,
                        help="训练次数大小")
    parser.add_argument("--embeddings_lr",
                        default=1e-2,
                        type=float,
                        help="Embeddings初始学习步长")
    parser.add_argument("--encoder_lr",
                        default=1e-4,
                        type=float)
    parser.add_argument("--learning_rate",
                        default=1e-4,
                        type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--train_batch_size",
                        default=1,  # 默认8
                        type=int,
                        help="训练时batch大小")
    parser.add_argument("--max_text_len",
                        default=128,  # 默认128
                        type=int,
                        help="文本最大长度")
    parser.add_argument("--seed",
                        default=8,  # 默认8
                        type=int,
                        help="初始化时的随机数种子")

    # 不可调固定内容
    parser.add_argument("--train_data_dir",
                        default='data/train.txt',
                        type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--test_data_dir",
                        default='data/test.txt',
                        type=str)
    parser.add_argument("--config_mla1",
                        default='checkpoint/mla_config1.json',
                        type=str)
    parser.add_argument("--config_mla2",
                        default='checkpoint/mla_config2.json',
                        type=str)
    parser.add_argument("--config_mla3",
                        default='checkpoint/mla_config3.json',
                        type=str)
    parser.add_argument("--config_lstm1",
                        default='checkpoint/lstm_config1.json',
                        type=str)
    parser.add_argument("--model_pth",
                        default='zh_large_bright/',
                        type=str)
    parser.add_argument("--mla_dir1",
                        default='checkpoint/mla_crf1/',
                        type=str)
    parser.add_argument("--mla_dir2",
                        default='checkpoint/mla_crf2/',
                        type=str)
    parser.add_argument("--mla_dir3",
                        default='checkpoint/mla_crf3/',
                        type=str)
    parser.add_argument("--lstm_dir1",
                        default='checkpoint/lstm_crf1/',
                        type=str)
    parser.add_argument("--vocab_dir",
                        default='zh_large_bright/vocab.txt',
                        type=str,
                        help="The vocab data dir.")
    parser.add_argument("--START_TAG",
                        default="[CLS]",
                        type=str)
    parser.add_argument("--STOP_TAG",
                        default="[SEP]",
                        type=str)
    parser.add_argument("--MASK_TAG",
                        default="[MASK]",
                        type=str)
    parser.add_argument("--tag_to_ix",
                        default={"B-Time": 0, "I-Time": 1, "B-Location": 2, "I-Location": 3, "B-Object": 4,
                                 "I-Object": 5,
                                 "B-Participant": 6, "I-Participant": 7, "B-Means": 8, "I-Means": 9, "B-Denoter": 10,
                                 "I-Denoter": 11, "o": 12, "[CLS]": 13, "[SEP]": 14, "[MASK]": 15},
                        type=dict)
    parser.add_argument("--ix_to_tag",
                        default={0: "B-Time", 1: "I-Time", 2: "B-Location", 3: "I-Location", 4: "B-Object",
                                 5: "I-Object", 6: "B-Participant", 7: "I-Participant", 8: "B-Means",
                                 9: "I-Means", 10: "B-Denoter", 11: "I-Denoter",
                                 12: "o", 13: "[CLS]", 14: "[SEP]", 15: "[MASK]"},
                        type=dict)
    parser.add_argument("--tag_to_tag",
                        default={"B-Time": "Time", "I-Time": "Time", "B-Location": "Location",
                                 "I-Location": "Location", "B-Object": "Object", "I-Object": "Object",
                                 "B-Participant": "Participant", "I-Participant": "Participant",
                                 "B-Means": "Means", "I-Means": "Means", "B-Denoter": "Denoter",
                                 "I-Denoter": "Denoter", "o": "Other",
                                 "[CLS]": "Other", "[SEP]": "Other", "[MASK]": "Other"},
                        type=dict)
    parser.add_argument("--tag_to_idx",
                        default={"Other": 0, "Time": 1, "Location": 2, "Object": 3,
                                 "Participant": 4, "Means": 5, "Denoter": 6},
                        type=dict)
    parser.add_argument("--tag_to_score",
                        default={0: 1, 1: 1, 2: 1, 3: 1, 4: 1,
                                 5: 1, 6: 1, 7: 1, 8: 1,
                                 9: 1, 10: 10, 11: 10,
                                 12: 5, 13: 1, 14: 1, 15: 1},
                        type=dict)
    parser.add_argument("--ix_to_idx",
                        default={0: 1, 1: 1, 2: 2, 3: 2, 4: 3,
                                 5: 3, 6: 4, 7: 4, 8: 5,
                                 9: 5, 10: 6, 11: 6,
                                 12: 0, 13: 0, 14: 0, 15: 0},
                        type=dict)
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    my_logger1 = logging.getLogger(__name__)
    my_logger2 = Logger(logger=__name__)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataloader = get_dataloader(args.train_data_dir)
    # main_thread1 = threading.Thread(target=mla_train1, args=(args, my_logger1, dataloader))
    # main_thread2 = threading.Thread(target=mla_train2, args=(args, my_logger2, dataloader))
    # main_thread3 = threading.Thread(target=lstm_train1, args=(args, my_logger1, dataloader))
    # main_thread4 = threading.Thread(target=mla_train3, args=(args, my_logger2, dataloader))
    # main_thread1.start()
    # main_thread2.start()
    # main_thread3.start()
    # main_thread4.start()
    _ = mla_train1(args, my_logger1, dataloader)





