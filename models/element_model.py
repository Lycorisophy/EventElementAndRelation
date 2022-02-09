from models.rnn_pos_model import RelClassifyModel
from models.ner_model import *
from models.text_classify_models import *
from nn.encoder import BiEncoder as BE
from nn.encoder import SingleEncoder as SE


class MyElectraEmbeddings(MyModule):
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.Embedding(config.element_type_num, config.embedding_size, padding_idx=0)

    def forward(self, input):
        return self.embeddings(input)


class ElementEmbedding(MyModule):
    def __init__(self, config, args):
        super(ElementEmbedding, self).__init__()
        self.config = config
        self.embedding = MyElectraModel(config)
        self.self_encoder = SE(config.hidden_size,
                               config.num_hidden_layers,
                               config.num_attention_heads)

    def from_pretrained(self, filenames):
        self.embedding.from_pretrained(filenames, config=self.config)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, m=None):
        embeded = self.embedding(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
        return self.self_encoder(embeded, m)


class BaElementEmbedding(MyModule):
    def __init__(self, config, args):
        super(BaElementEmbedding, self).__init__()
        self.config = config
        self.embedding = MyElectraModel(config)
        self.self_encoder = BE(config.hidden_size,
                               args.max_sent_len,
                               config.num_hidden_layers,
                               config.num_attention_heads,
                               args.num_attention_heads)

    def from_pretrained(self, filenames):
        self.embedding.from_pretrained(filenames, config=self.config)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, m=None):
        embeded = self.embedding(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
        return self.self_encoder(embeded, m)


class BgElementEmbedding(MyModule):
    def __init__(self, config, args):
        super(BgElementEmbedding, self).__init__()
        self.config = config
        self.embedding = MyElectraModel(config)
        self.self_encoder = nn.GRU(config.hidden_size,
                                   config.hidden_size // 2,
                                   batch_first=True,
                                   num_layers=2,
                                   bidirectional=True)

    def from_pretrained(self, filenames):
        self.embedding.from_pretrained(filenames, config=self.config)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, m=None):
        embeded = self.embedding(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
        x, _ = self.self_encoder(embeded)
        return x


class BbElementEmbedding(MyModule):
    def __init__(self, config, args):
        super(BbElementEmbedding, self).__init__()
        self.config = config
        self.embedding = MyElectraModel(config)
        self.self_encoder1 = nn.GRU(config.hidden_size,
                                    config.hidden_size // 2,
                                    batch_first=True,
                                    num_layers=2,
                                    bidirectional=True)
        self.self_encoder2 = BE(config.hidden_size,
                                args.max_sent_len,
                                config.num_hidden_layers,
                                config.num_attention_heads,
                                args.num_attention_heads)

    def from_pretrained(self, filenames):
        self.embedding.from_pretrained(filenames, config=self.config)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, m=None):
        embeded = self.embedding(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
        x, _ = self.self_encoder1(embeded)
        return self.self_encoder2(x, m)


class BbConElementEmbedding(MyModule):
    def __init__(self, config, args):
        super(BbConElementEmbedding, self).__init__()
        self.config = config
        self.embedding = MyElectraModel(config)
        self.self_encoder1 = nn.GRU(config.hidden_size,
                                    config.hidden_size // 2,
                                    batch_first=True,
                                    num_layers=2,
                                    bidirectional=True)
        self.self_encoder2 = BE(config.hidden_size,
                                args.max_sent_len,
                                config.num_hidden_layers,
                                config.num_attention_heads,
                                args.num_attention_heads)

    def from_pretrained(self, filenames):
        self.embedding.from_pretrained(filenames, config=self.config)

    def forward(self, input_ids1, input_ids2, token_type_ids1=None, token_type_ids2=None,
                position_ids1=None, position_ids2=None):
        embeded1 = self.embedding(input_ids=input_ids1, token_type_ids=token_type_ids1, position_ids=position_ids1)
        embeded2 = self.embedding(input_ids=input_ids2, token_type_ids=token_type_ids2, position_ids=position_ids2)
        x, _ = self.self_encoder1(embeded1 + embeded2 / 2)
        return self.self_encoder2(x)


class BbConEleElementEmbedding(MyModule):
    def __init__(self, config, args):
        super(BbConEleElementEmbedding, self).__init__()
        self.config = config
        self.embedding = MyElectraModel(config)
        self.size = config.hidden_size
        self.ee = nn.Embedding(config.element_type_num, self.size//2)
        self.self_encoder1 = nn.GRU(self.size+self.size//2,
                                    self.size+self.size//2,
                                    batch_first=True,
                                    num_layers=2,
                                    bidirectional=True)
        self.dense = nn.Linear(self.size*3, self.size)
        self.self_encoder2 = BE(self.size,
                                args.max_sent_len,
                                config.num_hidden_layers,
                                config.num_attention_heads,
                                args.num_attention_heads)

    def from_pretrained(self, filenames):
        self.embedding.from_pretrained(filenames, config=self.config)

    def forward(self, input_ids1, input_ids2, ele, token_type_ids1=None, token_type_ids2=None,
                position_ids1=None, position_ids2=None):
        embeded1 = self.embedding(input_ids=input_ids1, token_type_ids=token_type_ids1, position_ids=position_ids1)
        embeded2 = self.embedding(input_ids=input_ids2, token_type_ids=token_type_ids2, position_ids=position_ids2)
        ee = self.ee(ele)
        x = embeded1 + embeded2 / 2
        x = torch.cat([x, ee], 2)
        x, _ = self.self_encoder1(x)
        x = self.dense(x)
        return self.self_encoder2(x)


class Element(nn.Module):
    def __init__(self, config, args):
        super(Element, self).__init__()
        self.args = args
        self.tag_to_ix = args.tag_to_ix
        self.tagset_size = len(self.tag_to_ix)
        self.fc1 = nn.Linear(in_features=config.hidden_size,
                             out_features=config.hidden_size * 2)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(in_features=config.hidden_size * 2,
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

    def get_result(self, x):
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
                except IndexError:
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
                except IndexError:
                    label[i][self.tagset_size - 1] = 100
        if t > 0.9:
            return self.soft(label * 0.9)
        if t < 0.1:
            return self.soft(label * 0.1)
        return self.soft(label * t)

    def test(self, x, y):
        x = self.get_result(x)
        acc = self.get_acc(x, y)
        return acc

    def get_guess(self, x):
        return self.get_result(x)

    def get_guess_acc(self, x, y):
        x = self.get_result(x)
        acc = self.get_acc(x, y)
        return x, acc

    def forward(self, x, y, t=1):
        x = self.drop(x)
        x = self.get_result(x)
        acc = self.get_acc(x, y)
        y = self._dynamic_target(x, y, t)
        y = self.soft(y)
        return self.loss1(x, y), acc
