from nn.role_pos_embeddings import MyElectraModel
from models.rnn_pos_model import *
from models.common_models import *


class MyModel(SuperRolePosModel):
    def __init__(self, config, args):
        super(MyModel, self).__init__(MyElectraModel, RelClassifyModel, config, args)

