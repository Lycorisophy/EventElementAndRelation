from models.common_models import *
from nn.role_pos_embeddings import MyElectraModel
from models.rnn_pos_model import MLRelClassifyModel


class MyModel(SuperRolePosModel):
    def __init__(self, config, args):
        super(MyModel, self).__init__(MyElectraModel, MLRelClassifyModel, config, args)
