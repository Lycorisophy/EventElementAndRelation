from models.common_models import *
from nn.role_embeddings import MyElectraModel
from models.rnn_cnn_model import RelClassifyModel


class MyModel(SuperRoleModel):
    def __init__(self, config, args):
        super(MyModel, self).__init__(MyElectraModel, RelClassifyModel, config, args)
