# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from openvqa.utils.make_mask import make_mask
from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm
from openvqa.models.mcan.mca import MCA_ED
from openvqa.models.mcan.adapter import Adapter
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

        return x_atted, att


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.__C = __C

        if self.__C.BERT_ENCODER:
            self.encoder = BertModel.from_pretrained(self.__C.BERT_VER)
        elif not self.__C.BERT_ENCODER and self.__C.USE_BERT:
            self.bert_layer = BertModel.from_pretrained(self.__C.BERT_VER)
            # Freeze BERT layers
            for param in self.bert_layer.parameters():
                param.requires_grad = False
        # Loading the GloVe embedding weights
        elif __C.USE_GLOVE:
            self.embedding = nn.Embedding(
                num_embeddings=token_size,
                embedding_dim=__C.WORD_EMBED_SIZE
            )
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.adapter = Adapter(__C)

        self.backbone = MCA_ED(__C)

        # Flatten to vector
        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)

        # Classification layers
        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)


    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix):

        # Pre-process Language Feature
        if self.__C.USE_BERT:
            lang_feat_mask = make_mask(ques_ix[:, 1:-1].unsqueeze(2))
        else:
            lang_feat_mask = make_mask(ques_ix.unsqueeze(2))

        if self.__C.BERT_ENCODER:
            outputs = self.encoder(ques_ix)
            last_hidden_state = outputs[0]
            lang_feat = last_hidden_state[:, 1:-1, :] # remove CLS and SEP, making this to max_token=14
        elif not self.__C.BERT_ENCODER and self.__C.USE_BERT:
            outputs = self.bert_layer(ques_ix)
            last_hidden_state = outputs[0][:, 1:-1, :] # remove CLS and SEP, making this to max_token=14
            lang_feat, _ = self.lstm(last_hidden_state)
        elif self.__C.USE_GLOVE:
            lang_feat = self.embedding(ques_ix)
            lang_feat, _ = self.lstm(lang_feat)

        img_feat, img_feat_mask = self.adapter(frcn_feat, grid_feat, bbox_feat)

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        # Flatten to vector
        lang_feat, _ = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat, img_att = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        # Classification layers
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat, img_att

