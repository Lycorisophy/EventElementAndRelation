# 事件关系分类模块
# 作者：宋杨
from models.nl2tensor import *
from utils.process_control import *
import os
import re
from utils.argutils import print_args
# from pathlib import Path
import argparse
import json
import torch.optim
import numpy as np
from tqdm import trange
from nn.configuration_electra import ElectraConfig
from models.my_optimizers import Ranger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from language_model.transformers import ElectraTokenizer
from models.element_model import BbConElementEmbedding as MyEmbedding
from relation_models.base_model import BaseModel as MyModel
import datetime
import matplotlib.pyplot as plt


# 设置全局变量
def set_args(filename):
    parser = argparse.ArgumentParser()
    # 可调参数
    parser.add_argument("--train_epochs",
                        default=30,  # 默认5
                        type=int,
                        help="训练次数大小")
    parser.add_argument("--seed",
                        default=14,
                        type=int,
                        help="初始化时的随机数种子")
    parser.add_argument("--embeddings_lr",
                        default=2e-4,
                        type=float,
                        help="Embeddings初始学习步长")
    parser.add_argument("--encoder_lr",
                        default=5e-4,
                        type=float)
    parser.add_argument("--learning_rate",
                        default=5e-4,
                        type=float)
    parser.add_argument("--mymodel_save_dir",
                        default='checkpoint/relation/element_eb/',
                        type=str)
    parser.add_argument("--embedding_name",
                        default='eb_embedding.bin',
                        type=str,
                        help="The output embedding model filename.")
    parser.add_argument("--model_name",
                        default='eb_model.bin',
                        type=str,
                        help="The output model filename.")
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--train_batch_size",
                        default=4,  # 默认8
                        type=int,
                        help="训练时batch大小")
    parser.add_argument("--max_sent_len",
                        default=128,  # 默认256
                        type=int,
                        help="文本最大长度")
    parser.add_argument("--num_attention_heads",
                        default=4,
                        type=int)
    parser.add_argument("--test_size",
                        default=.2,
                        type=float,
                        help="验证集大小")
    parser.add_argument("--all_data_dir",
                        default='data/RRC_data/all/',
                        type=str)
    parser.add_argument("--mymodel_config_dir",
                        default='config/relation_base_config.json',
                        type=str)
    parser.add_argument("--pretrained_model_dir",
                        default='checkpoint/event_element/',
                        type=str)
    parser.add_argument("--vocab_dir",
                        default='pretrained_model/pytorch_electra_180g_large/vocab.txt',
                        type=str,
                        help="The vocab data dir.")
    parser.add_argument("--rel2label",
                        default={'Causal': 0, 'Follow': 1, 'Accompany': 2, 'Concurrency': 3},
                        type=list)
    parser.add_argument("--tag_to_score",
                        default={0: 6, 1: 6, 2: 6, 3: 6},
                        type=list)
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="训练模式")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="验证模式")
    parser.add_argument("--no_gpu",
                        default=False,
                        action='store_true',
                        help="用不用gpu")
    parser.add_argument("--gradient_accumulation_steps",
                        default=1,
                        type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--optimize_on_cpu",
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU.")
    parser.add_argument("--fp16",
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit.")
    parser.add_argument("--loss_scale",
                        default=128,
                        type=float,
                        help="Loss scaling, positive power of 2 values can improve fp16 convergence.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true')
    args = parser.parse_args()
    print_args(args, parser)
    with open(filename, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    return args


# 设置全局环境
try:
    args = set_args('relation_classify_args.txt')
except FileNotFoundError:
    args = set_args('relation_classify_args.txt')
logger = get_logger()
set_environ()
today = datetime.datetime.now()
my_time = str(today.year)+'-'+str(today.month)+'-'+str(today.day)
if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    torch.distributed.init_process_group(backend='nccl')
loss_device = torch.device("cpu")
logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    device, n_gpu, bool(args.local_rank != -1), args.fp16))
if args.gradient_accumulation_steps < 1:
    raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
        args.gradient_accumulation_steps))


# 定义一个计算准确率的函数
# TODO(LySoY 2020年11月30日) 建议修改
def accuracy(preds, labels, seq_len):
    count, right = 0.1, 0.1
    for pred, label in zip(preds, labels):
        for i in range(seq_len):
            if label[i] != len(args.tag_to_ix) - 1 and label[i] != len(args.tag_to_ix) - 2 \
                    and label[i] != len(args.tag_to_ix) - 3 and label[i] != len(args.tag_to_ix) - 4:
                count += 1
                _, p = pred[i].topk(1)
                if int(label[i]) == p[0].item():
                    right += 1
    return right / count


# 关系转label
# TODO(LySoY 2020年11月30日) 不建议使用
def rel2label(t_label, args):
    try:
        return args.rel2label[t_label]
    except IndexError:
        return 0


def get_model(args, the_time=my_time):
    config = ElectraConfig.from_pretrained(args.mymodel_config_dir)
    # config_dict = config.to_dict()
    # print(config_dict)
    model = MyModel(config=config, args=args)
    if args.fp16:
        model.half()
    model.to(device)
    try:
        model.load(os.path.join(args.mymodel_save_dir, the_time+args.model_name))
    except OSError:
        print("PretrainedModelNotFound")
    return model


# 网络训练
def mymodel_train(args, logger, train_dataloader, test_dataloader):
    config = ElectraConfig.from_pretrained(args.mymodel_config_dir)
    config_dict = config.to_dict()
    print(config_dict)
    embedding = MyEmbedding(config=config, args=args)
    model = MyModel(config=config, args=args)
    if args.fp16:
        embedding.half()
        model.half()
    embedding.to(device)
    model.to(device)
    # model = get_model(args)
    embedding.from_pretrained(args.pretrained_model_dir + '30,12,1e-5,5e-5/2021-5-19ba_embedding.bin')
    param_optimizer1 = list(embedding.named_parameters())
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in param_optimizer1 if any(nd in n for nd in ['encoder'])],
         'weight_decay_rate': args.weight_decay,
         'lr': args.encoder_lr},
        {'params': [p for n, p in param_optimizer1 if any(nd in n for nd in ['embedding'])],
         'weight_decay_rate': args.weight_decay,
         'lr': args.embeddings_lr},
        {'params': [p for n, p in param_optimizer1 if all(nd not in n for nd in ['encoder', 'embedding'])],
         'weight_decay_rate': args.weight_decay,
         'lr': args.learning_rate},
    ]
    optimizer1 = Ranger(optimizer_grouped_parameters1)
    param_optimizer2 = list(model.named_parameters())
    optimizer_grouped_parameters2 = [
        {'params': [p for n, p in param_optimizer2],
         'weight_decay_rate': args.weight_decay,
         'lr': args.learning_rate},
    ]
    optimizer2 = Ranger(optimizer_grouped_parameters2)
    epochs = args.train_epochs
    loss_records, train_loss_set, acc_records, tacc_records = [], [], [], []
    # n_not_max, max_ttt_acc = 0, 0
    torch.backends.cudnn.benchmark = True
    for _ in trange(epochs, desc='Epochs'):
        eval_accuracy = 0
        train_loss, train_accuracy = 0, 0
        nb_train_steps = 0
        nb_eval_steps = 0

        embedding.train()
        model.train()
        for step, batch in enumerate(train_dataloader):
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            input1, input2, labels, ft1, ft2 = batch
            input1 = input1.permute(1, 0, 2)
            input2 = input2.permute(1, 0, 2)
            ft1 = ft1.permute(1, 0, 2)
            ft2 = ft2.permute(1, 0, 2)
            embeded1 = embedding(input1[0].long(), ft1[0].long(), input1[1].long(), ft1[1].long(), input1[2].long(),
                                 ft1[2].long())
            embeded2 = embedding(input2[0].long(), ft2[0].long(), input2[1].long(), ft2[1].long(), input2[2].long(),
                                 ft2[2].long())
            loss, tmp_train_accuracy = model(embeded1, embeded2, labels)
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer1.step()
            optimizer2.step()
            train_loss += loss.item()
            train_accuracy += tmp_train_accuracy
            nb_train_steps += 1
        adjust_learning_rate(optimizer1, 0.9)
        adjust_learning_rate(optimizer2, 0.9)

        embedding.eval()
        model.eval()
        tag_seqs = []
        label_seqs = []
        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input1, input2, labels, ft1, ft2 = batch
            input1 = input1.permute(1, 0, 2)
            input2 = input2.permute(1, 0, 2)
            ft1 = ft1.permute(1, 0, 2)
            ft2 = ft2.permute(1, 0, 2)
            embeded1 = embedding(input1[0].long(), ft1[0].long(), input1[1].long(), ft1[1].long(), input1[2].long(),
                                 ft1[2].long())
            embeded2 = embedding(input2[0].long(), ft2[0].long(), input2[1].long(), ft2[1].long(), input2[2].long(),
                                 ft2[2].long())
            with torch.no_grad():
                pred, tmp_eval_accuracy = model.get_guess_acc(embeded1, embeded2, labels)
            tag_seqs.append(pred)
            label_seqs.append(labels)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        target_size = len(args.rel2label)
        result = np.zeros([target_size, target_size])
        for label, tag_seq in zip(label_seqs, tag_seqs):
            for la, tag in zip(label, tag_seq):
                _, p = tag.topk(1)
                pre = p[0].item()
                true = la.item()
                result[true][pre] += 1
        print(result)
        print_prf(result)
        f1 = print_f1(result)

        try:
            tt_loss = train_loss / nb_train_steps
            tt_acc = train_accuracy / nb_train_steps
            train_loss_set.append(tt_loss)
            ttt_acc = eval_accuracy / nb_eval_steps
            # if ttt_acc > max_ttt_acc:
            #     max_ttt_acc = ttt_acc
            #     n_not_max = 0
            # else:
            #     n_not_max += 1
            # if n_not_max == 3:
            #     break
            logger.info('mymodel训练损失:{:.4f},准确率为：{:.2f}%,测试集准确率为：{:.2f}%,测试集f1为：{:.2f}%'
                        .format(tt_loss, 100 * tt_acc, 100 * ttt_acc, 100 * f1))
            acc_records.append(tt_acc)
            loss_records.append(np.mean(train_loss_set))
            tacc_records.append(ttt_acc)
        except ZeroDivisionError:
            logger.info("错误！请降低batch大小")
        embedding.save(os.path.join(args.mymodel_save_dir, my_time + args.embedding_name))
        model.save(os.path.join(args.mymodel_save_dir, my_time + args.model_name))

    # 绘制准确率与误差变化曲线并保存网络
    print("绘制误差与测试集准确率变化曲线")
    plt.plot(loss_records, label='Train Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.plot(tacc_records, label='Accuracy Change')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    return model


# 获取数据集
# TODO(LySoY) replace the function with united functions or classes
def get_dataloader(filenames):
    tokenizer = ElectraTokenizer.from_pretrained(args.vocab_dir)
    input1 = []
    input2 = []
    labels = []
    ft1 = []
    ft2 = []
    try:
        E1 = np.load(filenames + "e1.npy")
        E2 = np.load(filenames + "e2.npy")
        R = np.load(filenames + "r.npy")
        C1 = np.load(filenames + "c1.npy")
        C2 = np.load(filenames + "c2.npy")
    except:
        from data.get_relation_from_xml import get_all_rel_role_con
        data = get_all_rel_role_con('../data/CEC', tokenizer)
        E1 = data[0]
        E2 = data[1]
        R = data[4]
        C1 = data[5]
        C2 = data[6]
    for i in range(len(E1)):
        label = rel2label(R[i], args)
        input1.append(convert_single_with_pos(tokenizer, E1[i], args.max_sent_len))
        input2.append(convert_single_with_pos(tokenizer, E2[i], args.max_sent_len, False))
        ft1.append(convert_single_with_pos(tokenizer, C1[i], args.max_sent_len))
        ft2.append(convert_single_with_pos(tokenizer, C2[i], args.max_sent_len, False))
        labels.append(label)
    train_input1, validation_input1, train_input2, validation_input2, \
    train_labels, validation_labels, train_ft1, validation_ft1, train_ft2, validation_ft2 = \
        train_test_split(input1, input2, labels, ft1, ft2,
                         random_state=args.seed, test_size=args.test_size)

    # 将训练集tensor并生成dataloader
    train_input1 = torch.Tensor(train_input1)
    train_input2 = torch.Tensor(train_input2)
    train_labels = torch.LongTensor(train_labels)
    train_ft1 = torch.Tensor(train_ft1)
    train_ft2 = torch.Tensor(train_ft2)
    batch_size = args.train_batch_size
    train_data = TensorDataset(train_input1, train_input2, train_labels, train_ft1, train_ft2)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=batch_size)

    if args.test_size > 0:
        # 将验证集tensor并生成dataloader
        validation_inputs1 = torch.Tensor(validation_input1)
        validation_inputs2 = torch.Tensor(validation_input2)
        validation_labels = torch.LongTensor(validation_labels)
        validation_fts1 = torch.Tensor(validation_ft1)
        validation_fts2 = torch.Tensor(validation_ft2)
        validation_data = TensorDataset(validation_inputs1, validation_inputs2, validation_labels,
                                        validation_fts1, validation_fts2)
        validation_sampler = RandomSampler(validation_data)
        validation_dataloader = DataLoader(validation_data,
                                           sampler=validation_sampler,
                                           batch_size=batch_size)
        return train_dataloader, validation_dataloader
    else:
        return train_dataloader, 0


def main():
    train_dataloader, validation_dataloader = get_dataloader(args.all_data_dir)
    model = mymodel_train(args, logger, train_dataloader, validation_dataloader)


if __name__ == "__main__":
    main()
