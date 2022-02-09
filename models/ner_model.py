from .attention import BiDecoder as BD
from nn.encoder import BiEncoder as BE
from models.common_models import *
from nn.role_pos_embeddings import MyElectraModel


def accuracy(preds, labels, seq_len, tag_size):
    count, right = 0, 0
    for pred, label in zip(preds, labels):
        for i in range(seq_len):
            if label[i] != tag_size - 1 and label[i] != tag_size - 2 \
                    and label[i] != tag_size - 3:
                count += 1
                _, p = pred[i].topk(1)
                if int(label[i]) == p[0].item():
                    right += 1
    return right / count


def accuracy2(preds, labels, seq_len: int, tag_size: int) -> float:
    count, right = 0.1, 0.0
    for pred, label in zip(preds, labels):
        for i in range(seq_len):
            if label[i] != tag_size - 1 and label[i] != tag_size - 2 \
                    and label[i] != tag_size - 3 and label[i] != tag_size - 4:
                count += 1
                _, p = pred[i].topk(1)
                if int(label[i]) == p[0].item():
                    right += 1
    acc = right / count
    if acc > 0.99:
        return 0.99
    if acc < 0.2:
        return 0.2
    return acc


class MyLoss1(nn.Module):
    def __init__(self):
        super(MyLoss1, self).__init__()

    def forward(self, x, y):
        return -torch.sum(y * torch.log(x + 1e-10))


class MyLoss2(nn.Module):
    def __init__(self, tag_size):
        super(MyLoss2, self).__init__()
        self.tag_size = tag_size

    def forward(self, x):
        return -torch.sum(1 / self.tag_size * torch.log(x + 1e-10))


class LstmNer1(nn.Module):
    def __init__(self, config, args):
        super(LstmNer1, self).__init__()
        self.args = args
        self.tag_to_ix = args.tag_to_ix
        self.tagset_size = len(args.tag_to_ix)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, num_layers=2, bidirectional=True)
        self.dense = nn.Linear(config.hidden_size * 2, self.tagset_size)
        self.soft = nn.Softmax(dim=-1)
        self.loss = MyLoss1()

    def _one_hot(self, x, tags):
        ys = torch.zeros_like(x)
        for y, tag in zip(ys, tags):
            for i in range(self.args.max_text_len):
                y[i][tag[i]] = 1
        return ys

    def _hard_hot(self, x, tags):
        ys = torch.ones_like(x)
        for y, tag in zip(ys, tags):
            for i in range(self.args.max_text_len):
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
        acc1 = accuracy(x, tags.detach().cpu().numpy(), self.args.max_text_len, len(self.tag_to_ix))
        acc2 = accuracy2(x, tags.detach().cpu().numpy(), self.args.max_text_len, len(self.tag_to_ix))
        return self.loss(x, y, 1 - acc2), acc1


class MlaNer1(nn.Module):
    def __init__(self, config, args):
        super(MlaNer1, self).__init__()
        self.args = args
        self.tagset_size = len(args.tag_to_ix)
        self.encoder = BD(config.hidden_size,
                          args.max_text_len,
                          config.num_hidden_layers2,
                          config.num_attention_heads,
                          1)
        self.dense = nn.Linear(config.hidden_size * 2, self.tagset_size)
        self.soft = nn.Softmax(dim=-1)
        self.loss1 = MyLoss1()
        self.loss2 = MyLoss2(len(args.tag_to_ix))

    def _one_hot(self, x, tags):
        ys = torch.zeros_like(x)
        for y, tag in zip(ys, tags):
            for i in range(self.args.max_text_len):
                y[i][tag[i]] = 1
        return ys

    def _hard_hot(self, x, tags):
        ys = torch.ones_like(x)
        for y, tag in zip(ys, tags):
            for i in range(self.args.max_text_len):
                y[i][tag[i]] = 100
        return ys

    def _dynamic_target(self, x, tags, t):
        ys = torch.ones_like(x)
        for y, tag in zip(ys, tags):
            for i in range(self.args.max_text_len):
                tmp = tag[i]
                y[i][tmp] = self.args.tag_to_score[tmp.item()] ** t
        return ys

    def _mask_hot(self, x, tags, masks):
        ys = torch.zeros_like(x)
        for y, tag, mask in zip(ys, tags, masks):
            for i in range(self.args.max_text_len):
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
            extended_attention_mask = extended_attention_mask.permute(0, 2, 1) * attention_mask
            extended_attention_mask = extended_attention_mask.permute(0, 2, 1)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            x += extended_attention_mask
        x = self.soft(x)
        acc1 = accuracy(x, tags.detach().cpu().numpy(), self.args.max_text_len, len(self.tag_to_ix))
        # acc2 = accuracy2(x, tags.detach().cpu().numpy(), args.max_text_len)
        y = self._dynamic_target(x, tags, t)
        y = self.soft(y)
        loss1 = self.loss1(x, y)
        # loss2 = self.loss2(x)
        # loss = (1-acc2)*loss1+acc2*loss2
        return loss1, acc1


class NerModel(nn.Module):
    def __init__(self, config, args):
        super(NerModel, self).__init__()
        self.args = args
        self.tag_to_ix = args.tag_to_ix
        self.tagset_size = len(self.tag_to_ix)
        self.self_encoder = BE(config.hidden_size,
                               args.max_sent_len,
                               config.num_hidden_layers,
                               config.num_attention_heads,
                               args.num_attention_heads)
        self.fc1 = nn.Linear(in_features=config.hidden_size,
                             out_features=config.hidden_size*2)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(in_features=config.hidden_size*2,
                             out_features=self.tagset_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.soft = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(0.2)
        self.loss1 = MyLoss1()
        self.loss2 = MyLoss2(self.tagset_size)

    def set_loss_device(self, device):
        self.loss.to(device)

    def load(self, output_model_file):
        model_state_dict = torch.load(output_model_file)
        self.load_state_dict(model_state_dict)

    def save(self, output_model_file):
        model_to_save = self.module if hasattr(self, 'module') else self
        torch.save(model_to_save.state_dict(), output_model_file)

    def get_result(self, x, m=None):
        x = self.self_encoder(x, m)
        x = self.LayerNorm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.soft(x)
        return x

    def get_acc(self, x, y):
        count, right = 0, 0
        for pred, label in zip(x, y):
            for p, l in zip(pred, label):
                if l != self.tagset_size - 1 \
                        and l != self.tagset_size - 2 \
                        and l != self.tagset_size - 3:
                    count += 1
                    _, p = p.topk(1)
                    if int(l) == p[0].item():
                        right += 1
        return right / count

    def one_hot(self, y):
        size = y.size()[0]
        label = torch.zeros(size, self.tagset_size).to(y.device)
        for i in range(size):
            for j in range(self.tagset_size):
                try:
                    label[i][int(y[i])] = 1
                except:
                    label[i][self.tagset_size - 1] = 1
        return label

    def _dynamic_target(self, x, tags, t):
        ys = torch.ones_like(x)
        for y, tag in zip(ys, tags):
            for i in range(self.args.max_sent_len):
                tmp = tag[i]
                y[i][tmp] = self.args.tag_to_score[tmp.item()] ** t
        return ys

    def soft_target(self, y, t):
        size = y.size()[0]
        label = torch.ones(size, self.tagset_size).to(y.device)
        for i in range(size):
            for j in range(self.tagset_size):
                try:
                    label[i][int(y[i])] = 100
                except:
                    label[i][self.tagset_size - 1] = 100
        if t > 0.9:
            return self.soft(label * 0.9)
        elif t < 0.1:
            return self.soft(label * 0.1)
        else:
            return self.soft(label * t)

    def test(self, x, y, m=None):
        x = self.get_result(x, m)
        acc = self.get_acc(x, y)
        return acc

    def get_guess(self, x, m=None):
        return self.get_result(x, m)

    def get_guess_acc(self, x, y, m=None):
        x = self.get_result(x, m)
        acc = self.get_acc(x, y)
        return x, acc

    def forward(self, x, y, m=None, t=1):
        x = self.drop(x)
        x = self.get_result(x, m)
        acc = self.get_acc(x, y)
        y = self._dynamic_target(x, y, t)
        y = self.soft(y)
        return 0.8*self.loss1(x, y)+0.2*self.loss2(x), acc


class MyNerModel(MyModule):
    def __init__(self, MyElectraModel, NerModel, config, args):
        super(MyNerModel, self).__init__()
        self.config = config
        self.embedding = MyElectraModel(config)
        self.ner_encoder = NerModel(config, args)

    def from_pretrained(self, filenames):
        self.embedding.from_pretrained(filenames, config=self.config)

    def get_text_emb(self, input_ids, token_type_ids=None, position_ids=None):
        text_embedding = self.embedding(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
        return text_embedding

    def test(self, input_ids, labels, token_type_ids=None, position_ids=None, input_mask=None):
        text_embedding = self.get_text_emb(input_ids, token_type_ids, position_ids)
        return self.ner_encoder.test(text_embedding, labels, input_mask)

    def get_guess(self, input_ids, token_type_ids=None, position_ids=None, input_mask=None):
        text_embedding = self.get_text_emb(input_ids, token_type_ids, position_ids)
        return self.ner_encoder.get_guess(text_embedding, input_mask)

    def get_guess_acc(self, input_ids, labels, token_type_ids=None, position_ids=None, input_mask=None):
        text_embedding = self.get_text_emb(input_ids, token_type_ids, position_ids)
        return self.ner_encoder.get_guess_acc(text_embedding, labels, input_mask)

    def forward(self, input_ids, labels, token_type_ids=None, position_ids=None, input_mask=None):
        text_embedding = self.get_text_emb(input_ids, token_type_ids, position_ids)
        return self.ner_encoder(text_embedding, labels, input_mask)


class SuperNerModel(NerModel):
    def __init__(self, config, args):
        super(SuperNerModel, self).__init__(MyElectraModel, NerModel, config, args)
