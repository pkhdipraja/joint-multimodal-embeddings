# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.data.ans_punct import prep_ans
import numpy as np
import random, re, json
#import en_vectors_web_lg, random, re, json
import torch


def shuffle_list(ans_list):
    random.shuffle(ans_list)


# ------------------------------
# ---- Initialization Utils ----
# ------------------------------

def img_feat_path_load(path_list):
    iid_to_path = {}

    for ix, path in enumerate(path_list):
        iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
        iid_to_path[iid] = path

    return iid_to_path


def img_feat_load(path_list):
    iid_to_feat = {}

    for ix, path in enumerate(path_list):
        iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
        img_feat = np.load(path)
        img_feat_x = img_feat['x'].transpose((1, 0))
        iid_to_feat[iid] = img_feat_x
        print('\rPre-Loading: [{} | {}] '.format(ix, path_list.__len__()), end='          ')

    return iid_to_feat


def ques_load(ques_list):
    qid_to_ques = {}

    for ques in ques_list:
        qid = str(ques['question_id'])
        qid_to_ques[qid] = ques

    return qid_to_ques


def tokenize(stat_ques_list, tokenizer, model, max_token, encoder_flag=False):
    if encoder_flag:
        token_to_ix = {}  # only for statistic!!

        for ques in stat_ques_list:
            words = re.sub(
                r"([.,'!?\"()*#:;])",
                '',
                ques['question'].lower()
            ).replace('-', ' ').replace('/', ' ')

            words = tokenizer.tokenize(words)
            ids = tokenizer.convert_tokens_to_ids(words)

            for tup in zip(words, ids):
                if tup[0] not in token_to_ix:
                    token_to_ix[tup[0]] = tup[1]
            
            pretrained_emb = None  # cannot use a fixed CWR as BERT is fine-tuned during training

    else:  # when using frozen BERT embeddings
        token_to_ix = {}

        for ques in stat_ques_list:
            words = re.sub(
                r"([.,'!?\"()*#:;])",
                '',
                ques['question']
            ).replace('-', ' ').replace('/', ' ')
            encoded_ques = tokenizer.encode_plus(
                            words,
                            add_special_tokens=True,
                            max_length=max_token,
                            pad_to_max_length=True,
                            return_tensors='pt',
                        )  
            indexed_tokens = tokenizer.convert_tokens_to_ids(encoded_ques)
            tokens_tensor = torch.tensor([indexed_tokens])
            
            # feed-forward operation
            model.eval()
            with torch.no_grad():
                outputs = model(tokens_tensor)
                hidden_states = outputs[2]
                # Concatenate the tensors for all layers.
                token_embeddings = torch.stack(hidden_states, dim=0)
                token_embeddings = torch.squeeze(token_embeddings, dim=1)
                token_embeddings = token_embeddings.permute(1, 0, 2)
            
                pretrained_emb = []
                
                for token in token_embeddings:
                    sum_vec = torch.sum(token[-4:],dim=0)
                    pretrained_emb.append(sum_vec)
                
            pretrained_emb = torch.stack(pretrained_emb)
            pretrained_emb = pretrained_emb.numpy()
        
            words = tokenizer.tokenize(words)
            for word in words:
                if word not in token_to_ix:
                    token_to_ix[word] = len(token_to_ix)

    return token_to_ix, pretrained_emb


# def ans_stat(stat_ans_list, ans_freq):
#     ans_to_ix = {}
#     ix_to_ans = {}
#     ans_freq_dict = {}
#
#     for ans in stat_ans_list:
#         ans_proc = prep_ans(ans['multiple_choice_answer'])
#         if ans_proc not in ans_freq_dict:
#             ans_freq_dict[ans_proc] = 1
#         else:
#             ans_freq_dict[ans_proc] += 1
#
#     ans_freq_filter = ans_freq_dict.copy()
#     for ans in ans_freq_dict:
#         if ans_freq_dict[ans] <= ans_freq:
#             ans_freq_filter.pop(ans)
#
#     for ans in ans_freq_filter:
#         ix_to_ans[ans_to_ix.__len__()] = ans
#         ans_to_ix[ans] = ans_to_ix.__len__()
#
#     return ans_to_ix, ix_to_ans


def ans_stat(json_file):
    ans_to_ix, ix_to_ans = json.load(open(json_file, 'r'))

    return ans_to_ix, ix_to_ans


# ------------------------------------
# ---- Real-Time Processing Utils ----
# ------------------------------------

def proc_img_feat(img_feat, img_feat_pad_size):
    if img_feat.shape[0] > img_feat_pad_size:
        img_feat = img_feat[:img_feat_pad_size]

    img_feat = np.pad(
        img_feat,
        ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
        mode='constant',
        constant_values=0
    )

    return img_feat


def proc_ques(ques, token_to_ix, max_token, tokenizer):

    words = re.sub(
        r"([.,'!?\"()*#:;])",
        '',
        ques['question'].lower()
    ).replace('-', ' ').replace('/', ' ')

    encoded_dict = tokenizer.encode_plus(
                        words,
                        add_special_tokens=True,
                        max_length=max_token,  # Pad & truncate all questions
                        pad_to_max_length=True,
                        return_tensors='pt',
                    )
    ques_ix = encoded_dict['input_ids']
    ques_ix = torch.squeeze(ques_ix)

    return ques_ix


def get_score(occur):
    if occur == 0:
        return .0
    elif occur == 1:
        return .3
    elif occur == 2:
        return .6
    elif occur == 3:
        return .9
    else:
        return 1.


def proc_ans(ans, ans_to_ix):
    ans_score = np.zeros(ans_to_ix.__len__(), np.float32)
    ans_prob_dict = {}

    for ans_ in ans['answers']:
        ans_proc = prep_ans(ans_['answer'])
        if ans_proc not in ans_prob_dict:
            ans_prob_dict[ans_proc] = 1
        else:
            ans_prob_dict[ans_proc] += 1

    for ans_ in ans_prob_dict:
        if ans_ in ans_to_ix:
            ans_score[ans_to_ix[ans_]] = get_score(ans_prob_dict[ans_])

    return ans_score

