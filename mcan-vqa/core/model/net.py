# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm
from core.model.mca import MCA_ED
from transformers import BertModel

import torch.nn as nn
import torch.nn.functional as F
import torch


# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()

        if __C.BERT_ENCODER:
            self.bert_encode = True
<<<<<<< HEAD
            self.encoder = BertModel.from_pretrained('bert-large-uncased')
        else:
            self.bert_encode = False
            self.bert_layer = BertModel.from_pretrained('bert-large-uncased') ###
=======
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
        else:
            self.bert_encode = False
            self.bert_layer = BertModel.from_pretrained('bert-base-uncased') ###
>>>>>>> 1ac1dec82e56c82ea13d39f9091a55633beecb06
            # freeze BERT layers
            for p in self.bert_layer.parameters():
                p.requires_grad = False
       #     self.embedding = nn.Embedding(
        #        num_embeddings=token_size,
         #       embedding_dim=__C.WORD_EMBED_SIZE
          #  )


        # Loading the GloVe embedding weights 
        # if __C.USE_GLOVE:
        #     self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
<<<<<<< HEAD
=======

     #   if __C.USE_BERT:
      #      self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
>>>>>>> 1ac1dec82e56c82ea13d39f9091a55633beecb06
        
        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.img_feat_linear = nn.Linear(
            __C.IMG_FEAT_SIZE,
            __C.HIDDEN_SIZE
        )

        self.backbone = MCA_ED(__C)

        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)

        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

    def forward(self, img_feat, ques_ix):
        # Make mask
        lang_feat_mask = self.make_mask(ques_ix[:, 1:-1].unsqueeze(2))
        img_feat_mask = self.make_mask(img_feat)

        if self.bert_encode:
            # ensure hidden state DIM is correct / change all to 768 or 1024
            # re-format to match lstm output, use torch.view()
            outputs = self.encoder(ques_ix)
            last_hidden_state = outputs[0]
            lang_feat = last_hidden_state[:, 1:-1, :]  # remove CLS and SEP, making this to MAX_TOKEN = 14
        else:
            # Pre-process Language Feature
            outputs = self.bert_layer(ques_ix) ###
            last_hidden_state = outputs[0] ###
<<<<<<< HEAD
            last_reshape = last_hidden_state[:, 1:-1, :] ###
            lang_feat, _ = self.lstm(last_reshape) ###
=======
            lang_feat, _ = self.lstm(last_hidden_state) ###
>>>>>>> 1ac1dec82e56c82ea13d39f9091a55633beecb06

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = torch.sigmoid(self.proj(proj_feat))

        return proj_feat


    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)


# class BertMCA(nn.Module):
#     def __init__(self, config, __C, pretrained_emb, token_size, answer_size, bertmodel):
#         super().__init__()
#         self.bert = bertmodel
#         self.config = config
#         self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
#         self.network = Net(__C, pretrained_emb, token_size, answer_size)

#     # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
#     # @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="bert-base-uncased")
#     def forward(
#         self,
#         img_feat,
#         input_ids
#     ):  
#         outputs = self.bert(input_ids)

#         ques_ix = input_ids[:, 1:-1]
#         lang_feat = outputs[0][:, 1:-1, :]

#         proj_feat = self.network(img_feat, ques_ix, lang_feat)
#         return proj_feat
