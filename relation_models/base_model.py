from nn.encoder import DotEncoder as DE
from utils.process_control import label_from_output
from models.my_loss_functions import *
from models.common_models import MyModule


class BaseModel(MyModule):
    def __init__(self, config, args):
        super(BaseModel, self).__init__()
        self.args = args
        self.tagset_size = len(args.rel2label)
        self.dot_encoder = DE(config.hidden_size,
                              config.intermediate_size,
                              config.num_hidden_layers,
                              config.num_attention_heads)
        self.fc1 = nn.Linear(in_features=config.hidden_size,
                             out_features=1)
        self.fc2 = nn.Linear(in_features=args.max_sent_len,
                             out_features=self.tagset_size)
        self.drop = nn.Dropout(0.2)
        self.soft = nn.Softmax(dim=-1)
        self.loss = CrossEntropyLoss()

    def set_loss_device(self, device):
        self.loss.to(device)

    def get_result(self, x1, x2):
        x = self.dot_encoder(x1, x2)
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
            except IndexError:
                pass
        return is_right / size

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
        elif t < 0.1:
            return self.soft(label * 0.1)
        return self.soft(label * t)

    def _dynamic_target(self, x, tags, t):
        ys = torch.ones_like(x)
        for y, tag in zip(ys, tags):
            y[tag] = self.args.tag_to_score[tag.item()] ** t
        return ys

    def test(self, x1, x2, y):
        x = self.get_result(x1, x2)
        acc = self.get_acc(x, y)
        return acc

    def get_guess(self, x1, x2):
        return self.get_result(x1, x2)

    def get_guess_acc(self, x1, x2, y):
        x = self.get_result(x1, x2)
        acc = self.get_acc(x, y)
        return x, acc

    def forward(self, x1, x2, y, t=1):
        x1 = self.drop(x1)
        x2 = self.drop(x2)
        x = self.get_result(x1, x2)
        acc = self.get_acc(x, y)
        y = self._dynamic_target(x, y, t)
        y = self.soft(y)
        return self.loss(x, y), acc