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
from language_model.transformers.configuration_electra import ElectraConfig
from models.my_optimizers import Ranger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from language_model.transformers import ElectraTokenizer
from models.small_model import MyModel
import datetime


# 设置全局变量
def set_args(filename):
    parser = argparse.ArgumentParser()
    # 可调参数
    parser.add_argument("--train_epochs",
                        default=20,  # 默认5
                        type=int,
                        help="训练次数大小")
    parser.add_argument("--embeddings_lr",
                        default=5e-4,
                        type=float,
                        help="Embeddings初始学习步长")
    parser.add_argument("--encoder_lr",
                        default=5e-4,
                        type=float)
    parser.add_argument("--learning_rate",
                        default=5e-4,
                        type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--train_batch_size",
                        default=16,  # 默认8
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
    parser.add_argument("--model_name",
                        default='small_model.bin',
                        type=str,
                        help="The output model filename.")
    parser.add_argument("--all_data_filename",
                        default='data/rel_data/all.csv',
                        type=str)
    parser.add_argument("--train_data_filename",
                        default='data/rel_data/train.csv',
                        type=str,
                        help="The input data filename. Should contain the .csv files (or other data files) for the "
                             "task.")
    parser.add_argument("--test_data_filename",
                        default='data/rel_data/test.csv',
                        type=str)
    parser.add_argument("--train_data_dir",
                        default='data/rel_data/',
                        type=str,
                        help="The input data dir.")
    parser.add_argument("--test_data_dir",
                        default='data/rel_data/',
                        type=str)
    parser.add_argument("--mymodel_config_dir",
                        default='config/relation_classify_config.json',
                        type=str)
    parser.add_argument("--mymodel_save_dir",
                        default='checkpoint/relation_classify/',
                        type=str)
    parser.add_argument("--pretrained_model_dir",
                        default='pretrained_model/pytorch_electra_180g_large/',
                        type=str)
    parser.add_argument("--vocab_dir",
                        default='pretrained_model/pytorch_electra_180g_large/vocab.txt',
                        type=str,
                        help="The vocab data dir.")
    parser.add_argument("--rel2label",
                        default={'Causal': 0, 'Follow': 1, 'Accompany': 2, 'Concurrency': 3, 'Other': 4},
                        type=dict)
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
    parser.add_argument("--seed",
                        default=6,
                        type=int,
                        help="初始化时的随机数种子")
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
    args = set_args('../config/relation_classify_args.txt')
except FileNotFoundError:
    args = set_args('../config/relation_classify_args.txt')
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
def rel2label(t_label, args):
    try:
        return args.rel2label[t_label]
    except:
        return len(args.rel2label)-1


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
def mymodel_train(args, logger, train_dataloader, validation_dataloader):
    config = ElectraConfig.from_pretrained(args.mymodel_config_dir)
    config_dict = config.to_dict()
    print(config_dict)
    model = MyModel(config=config, args=args)
    if args.fp16:
        model.half()
    model.to(device)
    # model = get_model(args)
    model.from_pretrained(args.pretrained_model_dir+'pytorch_model.bin')
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in ['embedding'])],
         'weight_decay_rate': args.weight_decay,
         'lr': args.embeddings_lr},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in ['encoder'])],
         'weight_decay_rate': args.weight_decay,
         'lr': args.encoder_lr},
    ]
    optimizer = Ranger(optimizer_grouped_parameters)
    epochs = args.train_epochs
    model.train()
    bio_records, train_loss_set, acc_records = [], [], []
    torch.backends.cudnn.benchmark = True
    for _ in trange(epochs, desc='Epochs'):
        tr_loss = 0
        eval_loss, eval_accuracy = 0, 0
        nb_tr_steps = 0
        nb_eval_steps = 0
        tmp_loss = []
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids1, input_ids2, labels = batch
            input_ids1 = input_ids1.squeeze(1).long()
            input_ids2 = input_ids2.squeeze(1).long()
            loss, tmp_eval_accuracy = model(input_ids1, input_ids2, labels)
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            tr_loss += loss.item()
            nb_tr_steps += 1
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
            tmp_loss.append(loss.item())
        adjust_learning_rate(optimizer, 0.9)
        try:
            train_loss_set.append(tr_loss / nb_tr_steps)
            logger.info('mymodel训练损失:{:.2f},准确率为：{:.2f}%'
                        .format(tr_loss / nb_tr_steps, 100 * eval_accuracy / nb_eval_steps))
            acc_records.append(eval_accuracy / nb_eval_steps)
            bio_records.append(np.mean(train_loss_set))
        except ZeroDivisionError:
            logger.info("错误！请降低batch大小")
        model.save(os.path.join(args.mymodel_save_dir, my_time+args.model_name))
    return model


# 网络测试
def mymodel_test(logger, test_dataloader, the_time=my_time):
    model = get_model(args, the_time)
    model.eval()
    acc_records = []
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps = 0
    for step, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids1, input_ids2, labels = batch
        input_ids1 = input_ids1.squeeze(1).long()
        input_ids2 = input_ids2.squeeze(1).long()
        with torch.no_grad():
            tmp_eval_accuracy = model.test(input_ids1, input_ids2, labels)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    try:
        logger.info('准确率为：{:.2f}%'
                    .format(100 * eval_accuracy / nb_eval_steps))
        acc_records.append(eval_accuracy / nb_eval_steps)
    except ZeroDivisionError:
        logger.info("错误！请降低batch大小")
    return acc_records


def mymodel_cal(logger, test_dataloader, the_time=my_time):
    model = get_model(args, the_time)
    model.eval()
    target_size = len(args.rel2label)
    result = np.zeros([target_size, target_size])
    for step, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids1, input_ids2, labels = batch
        input_ids1 = input_ids1.squeeze(1).long()
        input_ids2 = input_ids2.squeeze(1).long()
        with torch.no_grad():
            pred = model.get_guess(input_ids1, input_ids2)
        size = pred.size()[0]
        for i in range(size):
            try:
                result[labels[i], label_from_output(pred[i])] += 1
            except:
                continue
    print(result)
    return result


# 获取数据集
def get_dataloader(filename):
    tokenizer = ElectraTokenizer.from_pretrained(args.vocab_dir)
    input_ids1 = []
    input_ids2 = []
    labels = []
    cnt = 0
    Text1 = re.compile('.+\$')
    Text2 = re.compile('\$.+@')
    Text3 = re.compile('@.+')
    for line in read_lines(filename):
        line = re.sub(u'\t', '', line)
        text1 = re.sub(u'\$', '', Text1.findall(line)[0])
        text2 = re.sub(u'@', '', re.sub(u'\$', '', Text2.findall(line)[0]))
        t_label = re.sub(u'@', '', Text3.findall(line)[0])
        tmp1, _, _ = text2ids(tokenizer, text1, args.max_sent_len)
        tmp2, _, _ = text2ids(tokenizer, text2, args.max_sent_len)
        label = rel2label(t_label, args)
        input_ids1.append(tmp1)
        input_ids2.append(tmp2)
        labels.append(label)
        cnt += 1
    train_input1, validation_input1, train_input2, validation_input2, train_labels, validation_labels = \
        train_test_split(input_ids1, input_ids2, labels, random_state=args.seed, test_size=args.test_size)

    # 将训练集tensor并生成dataloader
    train_inputs1 = torch.Tensor(train_input1)
    train_inputs2 = torch.Tensor(train_input2)
    train_labels = torch.LongTensor(train_labels)
    batch_size = args.train_batch_size
    train_data = TensorDataset(train_inputs1, train_inputs2, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=batch_size)

    if args.test_size > 0:
        # 将验证集tensor并生成dataloader
        validation_inputs1 = torch.Tensor(validation_input1)
        validation_inputs2 = torch.Tensor(validation_input2)
        validation_labels = torch.LongTensor(validation_labels)
        validation_data = TensorDataset(validation_inputs1, validation_inputs2, validation_labels)
        validation_sampler = RandomSampler(validation_data)
        validation_dataloader = DataLoader(validation_data,
                                           sampler=validation_sampler,
                                           batch_size=batch_size)
        return train_dataloader, validation_dataloader
    else:
        return train_dataloader, _


def main():
    train_dataloader, validation_dataloader = get_dataloader(args.all_data_filename)
    model = mymodel_train(args, logger, train_dataloader, validation_dataloader)
    acc_records = mymodel_test(logger, validation_dataloader)
    result = mymodel_cal(logger, validation_dataloader)
    print_prf(result)


if __name__ == "__main__":
    main()
