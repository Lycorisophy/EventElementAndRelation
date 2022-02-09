from nn.encoder import BiEncoder as BE
from utils.process_control import label_from_output
from models.my_loss_functions import *
from nn.basic_model import *
from models.common_models import *


class TextClassifyModel(nn.Module):
    def __init__(self, config, args):
        super(TextClassifyModel, self).__init__()
        self.tagset_size = len(args.rel2label)
        self.self_encoder = BE(config.hidden_size,
                               args.max_sent_len,
                               config.num_hidden_layers,
                               config.num_attention_heads,
                               args.num_attention_heads)
        self.rnn = get_rnn(config)
        self.fc1 = nn.Linear(in_features=config.hidden_size,
                             out_features=1)
        self.fc2 = nn.Linear(in_features=args.max_sent_len,
                             out_features=self.tagset_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.soft = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(0.5)
        self.loss = CrossEntropyLoss()

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
        x, _ = self.rnn(x)
        x = self.fc1(x)
        x = x.squeeze(-1)
        x = self.fc2(x)
        x = self.soft(x)
        return x

    def get_acc(self, x, y):
        is_right = 0
        size = y.size()[0]
        for i in range(size):
            try:
                if y[i] == label_from_output(x[i]):
                    is_right += 1
            except:
                continue
        return is_right / size

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

    def forward(self, x, y, m=None):
        x = self.drop(x)
        x = self.get_result(x, m)
        acc = self.get_acc(x, y)
        y = self.one_hot(y)
        return self.loss(x, y), acc