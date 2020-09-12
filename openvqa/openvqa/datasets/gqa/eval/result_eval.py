# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from openvqa.datasets.gqa.eval.gqa_eval import GQAEval
import json, pickle
import numpy as np
import os


def eval(__C, dataset, ans_ix_list, pred_list, att_weights, result_eval_file, ensemble_file, log_file, valid=False):
    result_eval_file = result_eval_file + '.json'

    qid_list = [qid for qid in dataset.qid_list]
    ans_size = dataset.ans_size

    if __C.SPLIT['val'] in ['val', 'val_all']:
        # Object based features path and questions
        frcn_feat_path = __C.FEATS_PATH[__C.DATASET]['default-frcn']
        ques_file_path = __C.RAW_PATH[__C.DATASET][__C.SPLIT['val']]
        with open(ques_file_path, 'r') as f:
            ques_data = json.load(f)

        result = []
        for ix in range(len(qid_list)):
            result_buffer = {}
            attention = []

            result_buffer['questionId'] = qid_list[ix]
            result_buffer['prediction'] = dataset.ix_to_ans[str(ans_ix_list[ix])]
            # Load object features
            imageId = ques_data[str(qid_list[ix])]['imageId'] + ".npz"
            frcn_feat = np.load(os.path.join(frcn_feat_path, imageId), mmap_mode='r')
            bboxes = frcn_feat['bbox']
            img_width, img_height = frcn_feat['width'].item(), frcn_feat['height'].item()
            att_list = att_weights[ix, :].tolist()
            for num_bbox in range(bboxes.shape[0]):
                bbox = bboxes[num_bbox]
                # normalize bboxes
                x0, x1 = bbox[0].item()/img_width, bbox[2].item()/img_width
                y0, y1 = bbox[1].item()/img_height, bbox[3].item()/img_height
                attention.append([x0, y0, x1, y1, att_list[num_bbox]])
            result_buffer['attention'] = attention
            result.append(result_buffer)
    else:
        result = [{
            'questionId': qid_list[ix],
            'prediction': dataset.ix_to_ans[str(ans_ix_list[ix])],
        } for ix in range(len(qid_list))]

    print('Save the result to file: {}'.format(result_eval_file))
    json.dump(result, open(result_eval_file, 'w'))

    if __C.TEST_SAVE_PRED:
        print('Save the prediction vector to file: {}'.format(ensemble_file))

        pred_list = np.array(pred_list).reshape(-1, ans_size)
        result_pred = [{
            'pred': pred_list[qix],
            'qid': int(qid_list[qix])
        } for qix in range(qid_list.__len__())]
        pickle.dump(result_pred, open(ensemble_file, 'wb+'), protocol=-1)

    if valid:
        # create vqa object and vqaRes object
        ques_file_path = __C.RAW_PATH[__C.DATASET][__C.SPLIT['val']]
        choices_path = None
        if __C.SPLIT['val'] + '_choices' in __C.RAW_PATH[__C.DATASET]:
            choices_path = __C.RAW_PATH[__C.DATASET][__C.SPLIT['val'] + '_choices']

        eval_gqa = GQAEval(__C, result_eval_file, ques_file_path, choices_path, EVAL_CONSISTENCY=False)
        result_string, detail_result_string = eval_gqa.get_str_result()

        print('Write to log file: {}'.format(log_file))
        logfile = open(log_file, 'a+')

        for result_string_ in result_string:
            logfile.write(result_string_)
            logfile.write('\n')
            print(result_string_)

        for detail_result_string_ in detail_result_string:
            logfile.write(detail_result_string_)
            logfile.write("\n")

        logfile.write('\n')
        logfile.close()


