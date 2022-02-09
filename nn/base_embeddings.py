import os
import torch
import torch.nn as nn
from language_model.transformers.modeling_utils import PreTrainedModel
from language_model.transformers.configuration_electra import ElectraConfig
from language_model.transformers.utils import logging
logger = logging.get_logger(__name__)


class ElectraPreTrainedModel(PreTrainedModel):
    config_class = ElectraConfig
    base_model_prefix = "electra"
    authorized_missing_keys = [r"position_ids"]
    authorized_unexpected_keys = [r"electra\.embeddings_project\.weight", r"electra\.embeddings_project\.bias"]

    # Copied from transformers.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class MyElectraEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        input_shape = input_ids.size()
        device = input_ids.device
        if position_ids is None:
            position_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        return embeddings


class MyElectraModel(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = MyElectraEmbeddings(config)
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        embeddings = self.embeddings(input_ids, token_type_ids, position_ids)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
