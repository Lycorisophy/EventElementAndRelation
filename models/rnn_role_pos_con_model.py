from models.common_models import *
from nn.role_pos_embeddings import MyElectraModel
from models.rnn_pos_model import RelClassifyModel, attRelClassifyModel, gruRelClassifyModel, RelClassifyModel_bh


class MyModel(SuperRolePosConModel):
    def __init__(self, config, args):
        super(MyModel, self).__init__(MyElectraModel, RelClassifyModel, config, args)


class BlackHoleModel(SuperRolePosConModel):
    def __init__(self, config, args):
        super(BlackHoleModel, self).__init__(MyElectraModel, RelClassifyModel_bh, config, args)


class MyGruModel(SuperRolePosConModel):
    def __init__(self, config, args):
        super(MyGruModel, self).__init__(MyElectraModel, gruRelClassifyModel, config, args)


class MyAttModel(SuperRolePosConModel):
    def __init__(self, config, args):
        super(MyAttModel, self).__init__(MyElectraModel, attRelClassifyModel, config, args)