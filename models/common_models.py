import torch.nn as nn
import torch


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

    def load(self, output_model_file):
        model_state_dict = torch.load(output_model_file)
        self.load_state_dict(model_state_dict)

    def save(self, output_model_file):
        model_to_save = self.module if hasattr(self, 'module') else self
        torch.save(model_to_save.state_dict(), output_model_file)


class SuperModel(MyModule):
    def __init__(self, MyElectraModel, RelClassifyModel, config, args):
        super(SuperModel, self).__init__()
        self.config = config
        self.role = nn.Embedding(config.max_role_size, config.embedding_size)
        self.embedding = MyElectraModel(config)
        self.encoder = RelClassifyModel(config, args)

    def from_pretrained(self, filenames):
        self.embedding.from_pretrained(filenames, config=self.config)

    def test(self, input_ids1, input_ids2, labels):
        return self.encoder.test(self.embedding(input_ids=input_ids1), self.embedding(input_ids=input_ids2), labels)

    def get_guess(self, input_ids1, input_ids2):
        return self.encoder.get_guess(self.embedding(input_ids=input_ids1), self.embedding(input_ids=input_ids2))

    def forward(self, input_ids1, input_ids2, labels):
        return self.encoder.forward(self.embedding(input_ids=input_ids1), self.embedding(input_ids=input_ids2), labels)


class SuperRoleModel(MyModule):
    def __init__(self, MyElectraModel, RelClassifyModel, config, args):
        super(SuperRoleModel, self).__init__()
        self.config = config
        self.role = nn.Embedding(config.max_role_size, config.embedding_size)
        self.embedding = MyElectraModel(config)
        self.encoder = RelClassifyModel(config, args)

    def from_pretrained(self, filenames):
        self.embedding.from_pretrained(filenames, config=self.config)

    def get_emb(self, input_ids, role_ids=None):
        if role_ids is None:
            role_emb = self.role(torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device))
        else:
            role_emb = self.role(role_ids)
        text_embedding = self.embedding(input_ids=input_ids)
        return text_embedding + role_emb.detach()

    def test(self, input_ids1, input_ids2, labels, role_ids1=None, role_ids2=None):
        embedding1 = self.get_emb(input_ids1, role_ids1)
        embedding2 = self.get_emb(input_ids2, role_ids2)
        return self.encoder.test(embedding1, embedding2, labels)

    def get_guess(self, input_ids1, input_ids2, role_ids1=None, role_ids2=None):
        embedding1 = self.get_emb(input_ids1, role_ids1)
        embedding2 = self.get_emb(input_ids2, role_ids2)
        return self.encoder.get_guess(embedding1, embedding2)

    def forward(self, input_ids1, input_ids2, labels, role_ids1=None, role_ids2=None):
        embedding1 = self.get_emb(input_ids1, role_ids1)
        embedding2 = self.get_emb(input_ids2, role_ids2)
        return self.encoder.forward(embedding1, embedding2, labels)


class SuperRolePosModel(MyModule):
    def __init__(self, MyElectraModel, RelClassifyModel, config, args):
        super(SuperRolePosModel, self).__init__()
        self.config = config
        self.role = nn.Embedding(config.max_role_size, config.embedding_size)
        self.embedding = MyElectraModel(config)
        self.encoder = RelClassifyModel(config, args)

    def from_pretrained(self, filenames):
        self.embedding.from_pretrained(filenames, config=self.config)

    def get_emb(self, input_ids, token_type_ids=None, position_ids=None, role_ids=None):
        if role_ids is None:
            role_emb = self.role(torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device))
        else:
            role_emb = self.role(role_ids)
        text_embedding = self.embedding(input_ids, token_type_ids, position_ids)
        return text_embedding + role_emb.detach()

    def test(self, input_ids1, input_ids2, labels, token_type_ids1=None, token_type_ids2=None,
             position_ids1=None, position_ids2=None, role_ids1=None, role_ids2=None,
             input_mask1=None, input_mask2=None):
        embedding1 = self.get_emb(input_ids1, token_type_ids1, position_ids1, role_ids1)
        embedding2 = self.get_emb(input_ids2, token_type_ids2, position_ids2, role_ids2)
        return self.encoder.test(embedding1, embedding2, labels, input_mask1, input_mask2)

    def get_guess(self, input_ids1, input_ids2, token_type_ids1=None, token_type_ids2=None,
                  position_ids1=None, position_ids2=None, role_ids1=None, role_ids2=None,
                  input_mask1=None, input_mask2=None):
        embedding1 = self.get_emb(input_ids1, token_type_ids1, position_ids1, role_ids1)
        embedding2 = self.get_emb(input_ids2, token_type_ids2, position_ids2, role_ids2)
        return self.encoder.get_guess(embedding1, embedding2, input_mask1, input_mask2)

    def get_guess_acc(self, input_ids1, input_ids2, labels, token_type_ids1=None, token_type_ids2=None,
                      position_ids1=None, position_ids2=None, role_ids1=None, role_ids2=None,
                      input_mask1=None, input_mask2=None):
        embedding1 = self.get_emb(input_ids1, token_type_ids1, position_ids1, role_ids1)
        embedding2 = self.get_emb(input_ids2, token_type_ids2, position_ids2, role_ids2)
        return self.encoder.get_guess_acc(embedding1, embedding2, labels, input_mask1, input_mask2)

    def forward(self, input_ids1, input_ids2, labels, token_type_ids1=None, token_type_ids2=None,
                position_ids1=None, position_ids2=None, role_ids1=None, role_ids2=None,
                input_mask1=None, input_mask2=None):
        embedding1 = self.get_emb(input_ids1, token_type_ids1, position_ids1, role_ids1)
        embedding2 = self.get_emb(input_ids2, token_type_ids2, position_ids2, role_ids2)
        return self.encoder.forward(embedding1, embedding2, labels, input_mask1, input_mask2)


# TODO(LySoY 2020年11月30日) 建议继承上一个class
class SuperRolePosConModel(MyModule):
    def __init__(self, MyElectraModel, RelClassifyModel, config, args):
        super(SuperRolePosConModel, self).__init__()
        self.config = config
        self.role = nn.Embedding(config.max_role_size, config.embedding_size)
        self.embedding = MyElectraModel(config)
        self.encoder = RelClassifyModel(config, args)

    def from_pretrained(self, filenames):
        self.embedding.from_pretrained(filenames, config=self.config)

    def get_emb(self, input_ids, token_type_ids=None, position_ids=None, role_ids=None, con_ids=None):
        if role_ids is None:
            role_emb = self.role(torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device))
        else:
            role_emb = self.role(role_ids)
        if con_ids is None:
            con_emb = self.embedding(input_ids=torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device))
        else:
            con_emb = self.embedding(input_ids=con_ids)
        text_embedding = self.embedding(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
        return text_embedding + role_emb.detach()+con_emb/2

    def test(self, input_ids1, input_ids2, labels, token_type_ids1=None, token_type_ids2=None,
             position_ids1=None, position_ids2=None, ft_ids1=None, ft_ids2=None,
             input_mask1=None, input_mask2=None):
        embedding1 = self.get_emb(input_ids1, token_type_ids1, position_ids1, ft_ids1[0], ft_ids1[1])
        embedding2 = self.get_emb(input_ids2, token_type_ids2, position_ids2, ft_ids2[0], ft_ids2[1])
        return self.encoder.test(embedding1, embedding2, labels, input_mask1, input_mask2)

    def get_guess(self, input_ids1, input_ids2, token_type_ids1=None, token_type_ids2=None,
                  position_ids1=None, position_ids2=None, ft_ids1=None, ft_ids2=None,
                  input_mask1=None, input_mask2=None):
        embedding1 = self.get_emb(input_ids1, token_type_ids1, position_ids1, ft_ids1[0], ft_ids1[1])
        embedding2 = self.get_emb(input_ids2, token_type_ids2, position_ids2, ft_ids2[0], ft_ids2[1])
        return self.encoder.get_guess(embedding1, embedding2, input_mask1, input_mask2)

    def get_guess_acc(self, input_ids1, input_ids2, labels, token_type_ids1=None, token_type_ids2=None,
                      position_ids1=None, position_ids2=None, ft_ids1=None, ft_ids2=None,
                      input_mask1=None, input_mask2=None):
        embedding1 = self.get_emb(input_ids1, token_type_ids1, position_ids1, ft_ids1[0], ft_ids1[1])
        embedding2 = self.get_emb(input_ids2, token_type_ids2, position_ids2, ft_ids2[0], ft_ids2[1])
        return self.encoder.get_guess_acc(embedding1, embedding2, labels, input_mask1, input_mask2)

    def forward(self, input_ids1, input_ids2, labels, token_type_ids1=None, token_type_ids2=None,
                position_ids1=None, position_ids2=None, ft_ids1=None, ft_ids2=None,
                input_mask1=None, input_mask2=None):
        embedding1 = self.get_emb(input_ids1, token_type_ids1, position_ids1, ft_ids1[0], ft_ids1[1])
        embedding2 = self.get_emb(input_ids2, token_type_ids2, position_ids2, ft_ids2[0], ft_ids2[1])
        return self.encoder.forward(embedding1, embedding2, labels, input_mask1, input_mask2)


# class SuperNerModel(MyModule):
#     def __init__(self, MyElectraModel, NerModel, config, args):
#         super(SuperNerModel, self).__init__()
#         self.config = config
#         self.embedding = MyElectraModel(config)
#         self.ner_encoder = NerModel(config, args)
#
#     def from_pretrained(self, filenames):
#         self.embedding.from_pretrained(filenames, config=self.config)
#
#     def get_text_emb(self, input_ids, token_type_ids=None, position_ids=None):
#         text_embedding = self.embedding(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
#         return text_embedding
#
#     def test(self, input_ids, labels, token_type_ids=None, position_ids=None, input_mask=None):
#         text_embedding = self.get_text_emb(input_ids, token_type_ids, position_ids)
#         return self.ner_encoder.test(text_embedding, labels, input_mask)
#
#     def get_guess(self, input_ids, token_type_ids=None, position_ids=None, input_mask=None):
#         text_embedding = self.get_text_emb(input_ids, token_type_ids, position_ids)
#         return self.ner_encoder.get_guess(text_embedding, input_mask)
#
#     def get_guess_acc(self, input_ids, labels, token_type_ids=None, position_ids=None, input_mask=None):
#         text_embedding = self.get_text_emb(input_ids, token_type_ids, position_ids)
#         return self.ner_encoder.get_guess_acc(text_embedding, labels, input_mask)
#
#     def forward(self, input_ids, labels, token_type_ids=None, position_ids=None, input_mask=None):
#         text_embedding = self.get_text_emb(input_ids, token_type_ids, position_ids)
#         return self.ner_encoder(text_embedding, labels, input_mask)



# class SuperJointModel(MyModule):
#     def __init__(self, MyElectraModel, NerModel, RelClassifyModel, config, args):
#         super(SuperJointModel, self).__init__()
#         self.config = config
#         self.role = nn.Embedding(config.max_role_size, config.embedding_size)
#         self.embedding = MyElectraModel(config)
#         self.ner_encoder = NerModel(config, args)
#         self.rr_encoder = RelClassifyModel(config, args)
#
#     def from_pretrained(self, filenames):
#         self.embedding.from_pretrained(filenames, config=self.config)
#
#     def get_all_emb(self, input_ids, token_type_ids=None, position_ids=None, role_ids=None, con_ids=None):
#         if role_ids is None:
#             role_emb = self.role(torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device))
#         else:
#             role_emb = self.role(role_ids)
#         if con_ids is None:
#             con_emb = self.embedding(input_ids=torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device))
#         else:
#             con_emb = self.embedding(input_ids=con_ids)
#         text_embedding = self.embedding(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
#         return text_embedding + role_emb.detach()+con_emb/2
#
#     def get_other_emb(self, input_ids, role_ids=None, con_ids=None):
#         if role_ids is None:
#             role_emb = self.role(torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device))
#         else:
#             role_emb = self.role(role_ids)
#         if con_ids is None:
#             con_emb = self.embedding(input_ids=torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device))
#         else:
#             con_emb = self.embedding(input_ids=con_ids)
#         return role_emb.detach()+con_emb/2
#
#     def get_con_emb(self, input_ids, con_ids=None):
#         if con_ids is None:
#             con_emb = self.embedding(input_ids=torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device))
#         else:
#             con_emb = self.embedding(input_ids=con_ids)
#         return con_emb/2
#
#     def get_text_emb(self, input_ids, token_type_ids=None, position_ids=None):
#         text_embedding = self.embedding(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
#         return text_embedding
#
#     def test(self, input_ids1, input_ids2, labels, token_type_ids1=None, token_type_ids2=None,
#              position_ids1=None, position_ids2=None, ft_ids1=None, ft_ids2=None,
#              input_mask1=None, input_mask2=None, use_pipeline=False):
#         text_embedding1 = self.get_text_emb(input_ids1, token_type_ids1, position_ids1)
#         text_embedding2 = self.get_text_emb(input_ids2, token_type_ids2, position_ids2)
#         if use_pipeline:
#             role_emb1 = self.ner_encoder_encoder.get_guess_acc(text_embedding1, ft_ids1[0])
#             role_emb2 = self.ner_encoder_encoder.get_guess_acc(text_embedding2, ft_ids2[0])
#             other_embedding1 = self.get_con_emb(input_ids1, ft_ids1[1]) + role_emb1
#             other_embedding2 = self.get_con_emb(input_ids2, ft_ids2[1]) + role_emb2
#             return [role_emb1, role_emb2,
#                     self.rr_encoder.test(text_embedding1 + other_embedding1,
#                                                   text_embedding2 + other_embedding2,
#                                                   labels, input_mask1, input_mask2)]
#         other_embedding1 = self.get_other_emb(input_ids1, ft_ids1[0], ft_ids1[1])
#         other_embedding2 = self.get_other_emb(input_ids2, ft_ids2[0], ft_ids2[1])
#         return [self.ner_encoder.test(text_embedding1, ft_ids1[0]),
#                 self.ner_encoder.test(text_embedding2, ft_ids2[0]),
#                 self.rr_encoder.test(text_embedding1 + other_embedding1, text_embedding2 + other_embedding2,
#                                               labels, input_mask1, input_mask2)]
#
#     def get_guess(self, input_ids1, input_ids2, token_type_ids1=None, token_type_ids2=None,
#                   position_ids1=None, position_ids2=None, ft_ids1=None, ft_ids2=None,
#                   input_mask1=None, input_mask2=None, use_pipeline=False):
#         text_embedding1 = self.get_text_emb(input_ids1, token_type_ids1, position_ids1)
#         text_embedding2 = self.get_text_emb(input_ids2, token_type_ids2, position_ids2)
#         if use_pipeline:
#             role_emb1 = self.ner_encoder_encoder.get_guess(text_embedding1, ft_ids1[0])
#             role_emb2 = self.ner_encoder_encoder.get_guess(text_embedding2, ft_ids2[0])
#             other_embedding1 = self.get_con_emb(input_ids1, ft_ids1[1]) + role_emb1
#             other_embedding2 = self.get_con_emb(input_ids2, ft_ids2[1]) + role_emb2
#             return [role_emb1, role_emb2,
#                     self.rr_encoder.get_guess_acc(text_embedding1 + other_embedding1,
#                                                   text_embedding2 + other_embedding2,
#                                                   input_mask1, input_mask2)]
#         other_embedding1 = self.get_other_emb(input_ids1, ft_ids1[0], ft_ids1[1])
#         other_embedding2 = self.get_other_emb(input_ids2, ft_ids2[0], ft_ids2[1])
#         return [self.ner_encoder.get_guess(text_embedding1, ft_ids1[0]),
#                 self.ner_encoder.get_guess(text_embedding2, ft_ids2[0]),
#                 self.rr_encoder.get_guess(text_embedding1 + other_embedding1, text_embedding2 + other_embedding2,
#                                               input_mask1, input_mask2)]
#
#     def get_guess_acc(self, input_ids1, input_ids2, labels, token_type_ids1=None, token_type_ids2=None,
#                       position_ids1=None, position_ids2=None, ft_ids1=None, ft_ids2=None,
#                       input_mask1=None, input_mask2=None, use_pipeline=False):
#         text_embedding1 = self.get_text_emb(input_ids1, token_type_ids1, position_ids1)
#         text_embedding2 = self.get_text_emb(input_ids2, token_type_ids2, position_ids2)
#         if use_pipeline:
#             role_emb1 = self.ner_encoder_encoder.get_guess_acc(text_embedding1, ft_ids1[0])
#             role_emb2 = self.ner_encoder_encoder.get_guess_acc(text_embedding2, ft_ids2[0])
#             other_embedding1 = self.get_con_emb(input_ids1, ft_ids1[1]) + role_emb1[0]
#             other_embedding2 = self.get_con_emb(input_ids2, ft_ids2[1]) + role_emb2[0]
#             return [role_emb1, role_emb2,
#                     self.rr_encoder.get_guess_acc(text_embedding1 + other_embedding1, text_embedding2 + other_embedding2,
#                                     labels, input_mask1, input_mask2)]
#         other_embedding1 = self.get_other_emb(input_ids1, ft_ids1[0], ft_ids1[1])
#         other_embedding2 = self.get_other_emb(input_ids2, ft_ids2[0], ft_ids2[1])
#         return [self.ner_encoder.get_guess_acc(text_embedding1, ft_ids1[0]),
#                 self.ner_encoder.get_guess_acc(text_embedding2, ft_ids2[0]),
#                 self.rr_encoder.get_guess_acc(text_embedding1 + other_embedding1, text_embedding2 + other_embedding2,
#                                 labels, input_mask1, input_mask2)]
#
#
#     def forward(self, input_ids1, input_ids2, labels, token_type_ids1=None, token_type_ids2=None,
#                 position_ids1=None, position_ids2=None, ft_ids1=None, ft_ids2=None,
#                 input_mask1=None, input_mask2=None, use_pipeline=False):
#         text_embedding1 = self.get_text_emb(input_ids1, token_type_ids1, position_ids1)
#         text_embedding2 = self.get_text_emb(input_ids2, token_type_ids2, position_ids2)
#         if use_pipeline:
#             role_emb1 = self.ner_encoder_encoder(text_embedding1, ft_ids1[0])
#             role_emb2 = self.ner_encoder_encoder(text_embedding2, ft_ids2[0])
#             other_embedding1 = self.get_con_emb(input_ids1, ft_ids1[1]) + role_emb1[0]
#             other_embedding2 = self.get_con_emb(input_ids2, ft_ids2[1]) + role_emb2[0]
#             return [role_emb1, role_emb2,
#                     self.rr_encoder(text_embedding1 + other_embedding1, text_embedding2 + other_embedding2,
#                                     labels, input_mask1, input_mask2)]
#         other_embedding1 = self.get_other_emb(input_ids1, ft_ids1[0], ft_ids1[1])
#         other_embedding2 = self.get_other_emb(input_ids2, ft_ids2[0], ft_ids2[1])
#         return [self.ner_encoder(text_embedding1, ft_ids1[0]),
#                 self.ner_encoder(text_embedding2, ft_ids2[0]),
#                 self.rr_encoder(text_embedding1 + other_embedding1, text_embedding2 + other_embedding2,
#                                 labels, input_mask1, input_mask2)]
