from nn.role_embeddings import MyElectraModel
from models.base_model import RelClassifyModel
from models.common_models import *


class MyModel(SuperRoleModel):
    def __init__(self, config, args):
        super(MyModel, self).__init__(MyElectraModel, RelClassifyModel, config, args)
