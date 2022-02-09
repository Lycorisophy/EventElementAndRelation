from models.nl2tensor import *
from utils.process_control import *
import os
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
from models.ner_model import SuperNerModel as MyModel
import datetime
import matplotlib.pyplot as plt


# 设置全局参数
def set_args(filename):
    parser = argparse.ArgumentParser()
    # 可调参数
    parser.add_argument("--train_epochs",
                        default=20,  # 默认5
                        type=int,
                        help="训练次数大小")
    parser.add_argument("--seed",
                        default=10,
                        type=int,
                        help="初始化时的随机数种子")
    parser.add_argument("--embeddings_lr",
                        default=2e-5,
                        type=float,
                        help="Embeddings初始学习步长")
    parser.add_argument("--encoder_lr",
                        default=2e-5,
                        type=float)
    parser.add_argument("--mymodel_save_dir",
                        default='checkpoint/ner1/',
                        type=str)
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float)
    parser.add_argument("--model_name",
                        default='ner_model.bin',
                        type=str,
                        help="The output model filename.")
    parser.add_argument("--role_lr",
                        default=.0,
                        type=float,
                        help="Role_Embeddings初始学习步长")
    parser.add_argument("--weight_decay", default=.0, type=float)
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
    parser.add_argument("--all_data_dir",
                        default='data/RRC_data/all/',
                        type=str)
    parser.add_argument("--train_data_dir",
                        default='data/RRC_data/train/',
                        type=str,
                        help="The input data dir.")
    parser.add_argument("--test_data_dir",
                        default='data/RRC_data/test/',
                        type=str)
    parser.add_argument("--mymodel_config_dir",
                        default='config/ner_config.json',
                        type=str)
    parser.add_argument("--pretrained_model_dir",
                        default='pretrained_model/pytorch_electra_180g_large/',
                        type=str)
    parser.add_argument("--vocab_dir",
                        default='pretrained_model/pytorch_electra_180g_large/vocab.txt',
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
                        default={"o": 0, "B-Time": 1, "I-Time": 2, "B-Location": 3, "I-Location": 4, "B-Object": 5,
                                 "I-Object": 6,
                                 "B-Participant": 7, "I-Participant": 8, "B-Means": 9, "I-Means": 10, "B-Denoter": 11,
                                 "I-Denoter": 12, "[CLS]": 13, "[SEP]": 14, "[MASK]": 15},
                        type=list)
    parser.add_argument("--ix_to_tag",
                        default={0: "o", 1: "B-Time", 2: "I-Time", 3: "B-Location", 4: "I-Location", 5: "B-Object",
                                 6: "I-Object", 7: "B-Participant", 8: "I-Participant", 9: "B-Means",
                                 10: "I-Means", 11: "B-Denoter", 12: "I-Denoter",
                                 13: "[CLS]", 14: "[SEP]", 15: "[MASK]"},
                        type=list)
    parser.add_argument("--tag_to_tag",
                        default={"B-Time": "Time", "I-Time": "Time", "B-Location": "Location",
                                 "I-Location": "Location", "B-Object": "Object", "I-Object": "Object",
                                 "B-Participant": "Participant", "I-Participant": "Participant",
                                 "B-Means": "Means", "I-Means": "Means", "B-Denoter": "Denoter",
                                 "I-Denoter": "Denoter", "o": "Other",
                                 "[CLS]": "Other", "[SEP]": "Other", "[MASK]": "Other"},
                        type=list)
    parser.add_argument("--tag_to_idx",
                        default={"Other": 0, "Time": 1, "Location": 2, "Object": 3,
                                 "Participant": 4, "Means": 5, "Denoter": 6},
                        type=list)
    parser.add_argument("--tag_to_score",
                        default={0: 10, 1: 10, 2: 10, 3: 10, 4: 10,
                                 5: 10, 6: 10, 7: 10, 8: 10,
                                 9: 10, 10: 10, 11: 10,
                                 12: 10, 13: 10, 14: 10, 15: 10},
                        type=list)
    parser.add_argument("--max_role_size",
                        default=13,
                        type=int)
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
    args = set_args('relation_classify_rnn_role_pos_con_args.txt')
except FileNotFoundError:
    args = set_args('relation_classify_rnn_role_pos_con_args.txt')
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
    config_dict = config.to_dict()
    print(config_dict)
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
    model = MyModel(config=config, args=args)
    if args.fp16:
        model.half()
    model.to(device)
    # model = get_model(args)
    model.from_pretrained(args.pretrained_model_dir + 'pytorch_model.bin')
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in ['embedding'])],
         'weight_decay_rate': args.weight_decay,
         'lr': args.embeddings_lr},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in ['encoder'])],
         'weight_decay_rate': args.weight_decay,
         'lr': args.encoder_lr},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in ['role'])],
         'weight_decay_rate': args.weight_decay,
         'lr': args.role_lr},
        {'params': [p for n, p in param_optimizer if all(nd not in n for nd in ['embedding', 'encoder', 'role'])],
         'weight_decay_rate': args.weight_decay,
         'lr': args.encoder_lr},
    ]
    optimizer = Ranger(optimizer_grouped_parameters)
    epochs = args.train_epochs
    loss_records, train_loss_set, acc_records, tacc_records = [], [], [], []
    # n_not_max, max_ttt_acc = 0, 0
    torch.backends.cudnn.benchmark = True
    for _ in trange(epochs, desc='Epochs'):
        eval_accuracy = 0
        train_loss, train_accuracy = 0, 0
        nb_train_steps = 0
        nb_eval_steps = 0

        model.train()
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            input1, input_ft1 = batch
            input1 = input1.permute(1, 0, 2)
            loss, tmp_train_accuracy = model(input1[0].long(), input_ft1, input1[1].long(),
                                             input1[2].long())
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            train_loss += loss.item()
            train_accuracy += tmp_train_accuracy
            nb_train_steps += 1
        adjust_learning_rate(optimizer, 0.9)

        target_size = len(args.tag_to_ix)
        result = np.zeros([target_size, target_size])
        model.eval()
        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input1, input_ft1 = batch
            input1 = input1.permute(1, 0, 2)
            with torch.no_grad():
                pred, tmp_eval_accuracy = model.get_guess_acc(input1[0].long(), input1[1].long(),
                                             input1[2].long())
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
            size = pred.size()[0]
            for i in range(size):
                try:
                    result[input_ft1, label_from_output(pred[i])] += 1
                except:
                    continue
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


# 网络测试
def mymodel_test(logger, test_dataloader, the_time=my_time):
    model = get_model(args, the_time)
    model.eval()
    acc_records = []
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps = 0
    for step, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input1, input2, labels, input_ft1, input_ft2 = batch
        input1 = input1.permute(1, 0, 2)
        input2 = input2.permute(1, 0, 2)
        input_ft1 = input_ft1.permute(1, 0, 2)
        input_ft2 = input_ft2.permute(1, 0, 2)
        with torch.no_grad():
            tmp_eval_accuracy = model.test(input1[0].long(), input2[0].long(), labels, input1[1].long(),
                                           input1[1].long(), input1[2].long(), input2[2].long(),
                                           input_ft1, input_ft2)
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
        input1, input2, labels, input_ft1, input_ft2 = batch
        input1 = input1.permute(1, 0, 2)
        input2 = input2.permute(1, 0, 2)
        input_ft1 = input_ft1.permute(1, 0, 2)
        input_ft2 = input_ft2.permute(1, 0, 2)
        with torch.no_grad():
            pred = model.get_guess(input1[0].long(), input2[0].long(), input1[1].long(),
                                   input1[1].long(), input1[2].long(), input2[2].long(),
                                   input_ft1, input_ft2)
        size = pred.size()[0]
        for i in range(size):
            try:
                result[labels[i], label_from_output(pred[i])] += 1
            except:
                continue
    print(result)
    return result


# 获取数据集
def get_dataloader(filenames):
    tokenizer = ElectraTokenizer.from_pretrained(args.vocab_dir)
    input1 = []
    input_ft1 = []
    try:
        E1 = np.load(filenames+"e1.npy")
        E2 = np.load(filenames + "e2.npy")
        B1 = np.load(filenames+"b1.npy")
        B2 = np.load(filenames + "b2.npy")
        for b1, e1, b2, e2 in zip(B1, E1, B2, E2):
            input_ft1.append(convert_single_list(b1, args.max_sent_len, args.max_role_size))
            input1.append(convert_single_with_pos(tokenizer, e1, args.max_sent_len))
            input_ft1.append(convert_single_list(b2, args.max_sent_len, args.max_role_size))
            input1.append(convert_single_with_pos(tokenizer, e2, args.max_sent_len))

    except:
        from data.get_relation_from_xml import get_all_rel_role_con
        data = get_all_rel_role_con('../data/CEC', tokenizer)
        E1 = data[0] + data[1]
        B1 = data[2] + data[3]
        for b1, e1 in zip(B1, E1):
            input_ft1.append(convert_single_list(b1, args.max_sent_len, args.max_role_size))
            input1.append(convert_single_with_pos(tokenizer, e1, args.max_sent_len))


    # 将训练集tensor并生成dataloader
    train_input1, validation_input1, \
    input_ft1, validation_input_ft1 = \
        train_test_split(input1, input_ft1,
                         random_state=args.seed, test_size=args.test_size)
    train_input1 = torch.Tensor(train_input1)
    inputs_ft1 = torch.LongTensor(input_ft1)
    batch_size = args.train_batch_size
    train_data = TensorDataset(train_input1, inputs_ft1)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=batch_size)

    if args.test_size > 0:
        # 将验证集tensor并生成dataloader
        validation_inputs1 = torch.Tensor(validation_input1)
        validation_ft1 = torch.LongTensor(validation_input_ft1)
        validation_data = TensorDataset(validation_inputs1,
                                        validation_ft1)
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
    # acc_records = mymodel_test(logger, validation_dataloader)
    # result = mymodel_cal(logger, validation_dataloader)
    # print_prf(result)


if __name__ == "__main__":
    main()
