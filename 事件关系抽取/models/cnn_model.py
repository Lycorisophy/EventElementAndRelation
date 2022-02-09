from nn.encoder import BiEncoder as BE
from nn.encoder import DotEncoder as DE
from utils.process_control import label_from_output
from models.my_loss_functions import *
from nn.role_embeddings import MyElectraModel
from nn.basic_model import *
from models.common_models import *


# 事件关系分类网络定义
class RelClassifyModel(nn.Module):
    def __init__(self, config, args):
        super(RelClassifyModel, self).__init__()
        self.tagset_size = len(args.rel2label)
        self.self_encoder = BE(config.hidden_size,
                               args.max_sent_len,
                               config.num_hidden_layers,
                               config.num_attention_heads,
                               args.num_attention_heads)
        self.dot_encoder = DE(config.hidden_size,
                              config.intermediate_size,
                              config.num_hidden_layers,
                              config.num_attention_heads)
        self.cov = get_cnn(config)
        self.dense = nn.Linear(in_features=args.max_sent_len,
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

    def get_result(self, x1, x2):
        x1 = self.self_encoder(x1)
        x2 = self.self_encoder(x2)
        x = self.dot_encoder(x1, x2)
        x = self.LayerNorm(x)
        x = x.permute(0, 2, 1)
        x = self.cov(x)
        x = x.squeeze(-2)
        x = self.dense(x)
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

    def test(self, x1, x2, y):
        x = self.get_result(x1, x2)
        acc = self.get_acc(x, y)
        return acc

    def get_guess(self, x1, x2):
        return self.get_result(x1, x2)

    def forward(self, x1, x2, y):
        x1 = self.drop(x1)
        x2 = self.drop(x2)
        x = self.get_result(x1, x2)
        acc = self.get_acc(x, y)
        y = self.one_hot(y)
        return self.loss(x, y), acc


class MyModel(SuperModel):
    def __init__(self, config, args):
        super(MyModel, self).__init__(MyElectraModel, RelClassifyModel, config, args)