from models.rnn_pos_model import RelClassifyModel
from models.ner_model import *
from models.text_classify_models import *


class JointModel(MyModule):
    def __init__(self, MyElectraModel, NerModel, RelClassifyModel, config, args):
        super(JointModel, self).__init__()
        self.config = config
        self.role = nn.Embedding(config.max_role_size, config.embedding_size)
        self.embedding = MyElectraModel(config)
        self.ner_encoder = NerModel(config, args)
        self.rr_encoder = RelClassifyModel(config, args)

    def from_pretrained(self, filenames):
        self.embedding.from_pretrained(filenames, config=self.config)

    def get_all_emb(self, input_ids, token_type_ids=None, position_ids=None, role_ids=None, con_ids=None):
        if role_ids is None:
            role_emb = self.role(torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device))
        else:
            role_emb = self.role(role_ids)
        if con_ids is None:
            con_emb = self.embedding(input_ids=torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device))
        else:
            con_emb = self.embedding(input_ids=con_ids)
        text_embedding = self.embedding(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
        return text_embedding + role_emb.detach() + con_emb / 2

    def get_other_emb(self, input_ids, role_ids=None, con_ids=None):
        if role_ids is None:
            role_emb = self.role(torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device))
        else:
            role_emb = self.role(role_ids)
        if con_ids is None:
            con_emb = self.embedding(input_ids=torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device))
        else:
            con_emb = self.embedding(input_ids=con_ids)
        return role_emb.detach() + con_emb / 2

    def get_con_emb(self, input_ids, con_ids=None):
        if con_ids is None:
            con_emb = self.embedding(input_ids=torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device))
        else:
            con_emb = self.embedding(input_ids=con_ids)
        return con_emb / 2

    def get_text_emb(self, input_ids, token_type_ids=None, position_ids=None):
        text_embedding = self.embedding(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
        return text_embedding

    def test(self, input_ids1, input_ids2, labels, token_type_ids1=None, token_type_ids2=None,
             position_ids1=None, position_ids2=None, ft_ids1=None, ft_ids2=None,
             input_mask1=None, input_mask2=None, use_pipeline=False):
        text_embedding1 = self.get_text_emb(input_ids1, token_type_ids1, position_ids1)
        text_embedding2 = self.get_text_emb(input_ids2, token_type_ids2, position_ids2)
        if use_pipeline:
            role_emb1 = self.ner_encoder_encoder.get_guess_acc(text_embedding1, ft_ids1[0])
            role_emb2 = self.ner_encoder_encoder.get_guess_acc(text_embedding2, ft_ids2[0])
            other_embedding1 = self.get_con_emb(input_ids1, ft_ids1[1]) + role_emb1
            other_embedding2 = self.get_con_emb(input_ids2, ft_ids2[1]) + role_emb2
            return [role_emb1, role_emb2,
                    self.rr_encoder.test(text_embedding1 + other_embedding1,
                                         text_embedding2 + other_embedding2,
                                         labels, input_mask1, input_mask2)]
        other_embedding1 = self.get_other_emb(input_ids1, ft_ids1[0], ft_ids1[1])
        other_embedding2 = self.get_other_emb(input_ids2, ft_ids2[0], ft_ids2[1])
        return [self.ner_encoder.test(text_embedding1, ft_ids1[0]),
                self.ner_encoder.test(text_embedding2, ft_ids2[0]),
                self.rr_encoder.test(text_embedding1 + other_embedding1, text_embedding2 + other_embedding2,
                                     labels, input_mask1, input_mask2)]

    def get_guess(self, input_ids1, input_ids2, token_type_ids1=None, token_type_ids2=None,
                  position_ids1=None, position_ids2=None, ft_ids1=None, ft_ids2=None,
                  input_mask1=None, input_mask2=None, use_pipeline=False):
        text_embedding1 = self.get_text_emb(input_ids1, token_type_ids1, position_ids1)
        text_embedding2 = self.get_text_emb(input_ids2, token_type_ids2, position_ids2)
        if use_pipeline:
            role_emb1 = self.ner_encoder_encoder.get_guess(text_embedding1, ft_ids1[0])
            role_emb2 = self.ner_encoder_encoder.get_guess(text_embedding2, ft_ids2[0])
            other_embedding1 = self.get_con_emb(input_ids1, ft_ids1[1]) + role_emb1
            other_embedding2 = self.get_con_emb(input_ids2, ft_ids2[1]) + role_emb2
            return [role_emb1, role_emb2,
                    self.rr_encoder.get_guess_acc(text_embedding1 + other_embedding1,
                                                  text_embedding2 + other_embedding2,
                                                  input_mask1, input_mask2)]
        other_embedding1 = self.get_other_emb(input_ids1, ft_ids1[0], ft_ids1[1])
        other_embedding2 = self.get_other_emb(input_ids2, ft_ids2[0], ft_ids2[1])
        return [self.ner_encoder.get_guess(text_embedding1, ft_ids1[0]),
                self.ner_encoder.get_guess(text_embedding2, ft_ids2[0]),
                self.rr_encoder.get_guess(text_embedding1 + other_embedding1, text_embedding2 + other_embedding2,
                                          input_mask1, input_mask2)]

    def get_guess_acc(self, input_ids1, input_ids2, labels, token_type_ids1=None, token_type_ids2=None,
                      position_ids1=None, position_ids2=None, ft_ids1=None, ft_ids2=None,
                      input_mask1=None, input_mask2=None, use_pipeline=False):
        text_embedding1 = self.get_text_emb(input_ids1, token_type_ids1, position_ids1)
        text_embedding2 = self.get_text_emb(input_ids2, token_type_ids2, position_ids2)
        if use_pipeline:
            role_emb1 = self.ner_encoder_encoder.get_guess_acc(text_embedding1, ft_ids1[0])
            role_emb2 = self.ner_encoder_encoder.get_guess_acc(text_embedding2, ft_ids2[0])
            other_embedding1 = self.get_con_emb(input_ids1, ft_ids1[1]) + role_emb1[0]
            other_embedding2 = self.get_con_emb(input_ids2, ft_ids2[1]) + role_emb2[0]
            return [role_emb1, role_emb2,
                    self.rr_encoder.get_guess_acc(text_embedding1 + other_embedding1,
                                                  text_embedding2 + other_embedding2,
                                                  labels, input_mask1, input_mask2)]
        other_embedding1 = self.get_other_emb(input_ids1, ft_ids1[0], ft_ids1[1])
        other_embedding2 = self.get_other_emb(input_ids2, ft_ids2[0], ft_ids2[1])
        return [self.ner_encoder.get_guess_acc(text_embedding1, ft_ids1[0]),
                self.ner_encoder.get_guess_acc(text_embedding2, ft_ids2[0]),
                self.rr_encoder.get_guess_acc(text_embedding1 + other_embedding1, text_embedding2 + other_embedding2,
                                              labels, input_mask1, input_mask2)]

    def forward(self, input_ids1, input_ids2, labels, token_type_ids1=None, token_type_ids2=None,
                position_ids1=None, position_ids2=None, ft_ids1=None, ft_ids2=None,
                input_mask1=None, input_mask2=None, use_pipeline=False):
        text_embedding1 = self.get_text_emb(input_ids1, token_type_ids1, position_ids1)
        text_embedding2 = self.get_text_emb(input_ids2, token_type_ids2, position_ids2)
        if use_pipeline:
            role_emb1 = self.ner_encoder_encoder(text_embedding1, ft_ids1[0])
            role_emb2 = self.ner_encoder_encoder(text_embedding2, ft_ids2[0])
            other_embedding1 = self.get_con_emb(input_ids1, ft_ids1[1]) + role_emb1[0]
            other_embedding2 = self.get_con_emb(input_ids2, ft_ids2[1]) + role_emb2[0]
            return [role_emb1, role_emb2,
                    self.rr_encoder(text_embedding1 + other_embedding1, text_embedding2 + other_embedding2,
                                    labels, input_mask1, input_mask2)]
        other_embedding1 = self.get_other_emb(input_ids1, ft_ids1[0], ft_ids1[1])
        other_embedding2 = self.get_other_emb(input_ids2, ft_ids2[0], ft_ids2[1])
        return [self.ner_encoder(text_embedding1, ft_ids1[0]),
                self.ner_encoder(text_embedding2, ft_ids2[0]),
                self.rr_encoder(text_embedding1 + other_embedding1, text_embedding2 + other_embedding2,
                                labels, input_mask1, input_mask2)]


class SuperJointModel(JointModel):
    def __init__(self, config, args):
        super(SuperJointModel, self).__init__(MyElectraModel, NerModel, RelClassifyModel, config, args)
