# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm
from core.model.mca import MCA_ED
from transformers import BertModel, AlbertModel, BertPreTrainedModel
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable
from transformers.modeling_bert import BERT_INPUTS_DOCSTRING

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

        else:
            self.bert_encode = False
            self.embedding = nn.Embedding(
                num_embeddings=token_size,
                embedding_dim=__C.WORD_EMBED_SIZE
            )

        # self.encoder = BertModel.from_pretrained('bert-base-uncased')
        # self.encoder = AlbertModel.from_pretrained('albert-base-v2')
        # Loading the GloVe embedding weights 
        # if __C.USE_GLOVE:
        #     self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

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


    def forward(self, img_feat, ques_ix, lang_feat):
        #ques_ix = torch.squeeze(ques_ix)
        #att_mask = torch.squeeze(att_mask)
        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = self.make_mask(img_feat)

        # if self.bert_encode:
        #     # ensure hidden state DIM is correct / change all to 768 or 1024
        #     # re-format to match lstm output, use torch.view()
        #     outputs = self.encoder(ques_ix, att_mask)
        #     last_hidden_state = outputs[0]
        #     lang_feat = last_hidden_state[:, 1:-1, :]  # remove CLS and SEP, making this to MAX_TOKEN = 14
        # else:
        # Pre-process Language Feature
            # lang_feat = self.embedding(ques_ix)
            # lang_feat, _ = self.lstm(lang_feat)

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


class BertMCA(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.network = Net(config.__C, config.pretrained_emb, config.token_size, config.answer_size)

        self.init_weights()
   
    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    # @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="bert-base-uncased")
    def forward(
        self,
        img_feat,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )   

        ques_ix = input_ids[:, 1:-1]
        lang_feat = outputs[0][:, 1:-1, :]

        proj_feat = self.network(img_feat, ques_ix, lang_feat)
        return proj_feat