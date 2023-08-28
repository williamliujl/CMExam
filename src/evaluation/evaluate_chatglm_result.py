# -*- coding: utf-8 -*-

# --------------------------------------------
# @FileName: evaluate_chatglm_result.py
# @Author: ljl
# @Time: 2023/5/10
# @Description:
# --------------------------------------------

import os
import re
from transformers import AutoTokenizer, AutoModel
import argparse
import pandas as pd
from tqdm import tqdm

template_multi = "假设你是一位医疗行业专家，请回答下列问题。注意，该问题是多选题\n" \
                 "{}:\n{}\n" \
                 "注意，请给出两行，第一行只需要返回答案的英文选项，第二行进行简要的解释。输出格式限制为“答案：”，“解释：”"

template_single = "返回限制：只返回两行。" \
                  "假设你是一位医疗行业专家，请回答下列问题，注意是单选题，只需要返回一个最合适的选项。\n" \
                  "{}:\n{}\n" \
                  "注意，结果只有两行，第一行只需要返回答案的英文选项(注意只需要返回一个最合适的答案)，第二行进行简要的解释。输出格式限制为：“答案：”，“解释：”。\n" \
                  "注意，题目是单选题，若有多个合适的答案，只返回最准确的即可。"

def prediction(args):
    # load model
    tokenizer = AutoTokenizer.from_pretrained(args.modelpath, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.tokenizerpath, trust_remote_code=True).half().cuda()
    model = model.eval()

    def predict(data):
        results = []
        for content in tqdm(data):
            try:
                response, history = model.chat(tokenizer, content, history=[])
            except Exception as e:
                response = ""
            results.append(response)
        return results

    # load csv
    csv = pd.read_csv(args.filepath)
    questions = csv['Question'].values.tolist()
    options = csv['Options'].values.tolist()
    gt_answer = csv['Answer'].values.tolist()

    data = []
    raw_results = []
    for i in range(len(questions)):
        if len(gt_answer[i]) == 1:
            data.append(template_single.format(questions[i], options[i]))
        else:
            data.append(template_multi.format(questions[i], options[i]))

    raw_results.extend(predict(data))
    predicted_answer = []
    predicted_explanation = []
    for single in raw_results:
        try:
            answer = re.findall(r"答案：(.*)，", single)[0]
            exp = re.findall(r"解释：(.*)", single)[0]
            predicted_answer.append(answer)
            predicted_explanation.append(exp)
        except Exception as e:
            print(single, flush=True)
            predicted_answer.append("")
            predicted_explanation.append("")

    csv['raw_prediction'] = raw_results
    csv['predicted_answer'] = predicted_answer
    csv['predicted_explanation'] = predicted_explanation

    if not os.path.exists(args.savepath):
        os.mkdir(args.savepath)
    csv.to_csv(args.savepath, index=False)


def evaluation(args):
    csv = pd.read_csv(args.savepath)

    gt_exp = csv['Explanation'].values.tolist()
    predict_exp = csv['predicted_explanation'].values.tolist()
    # process pd.na
    gt_exp = [item if not pd.isna(item) else "" for item in gt_exp]
    predict_exp = [item if not pd.isna(item) else "" for item in predict_exp]

    gt_answer = csv['Answer'].values.tolist()
    predict_answer = csv['predicted_answer'].values.tolist()
    gt_answer_with_value = []
    predict_answer_with_value = []

    total = 0.0
    correct = 0.0
    for i in range(len(gt_answer)):
        if not pd.isna(predict_answer[i]):
            total += 1
            gt_answer_with_value.append(gt_answer[i])
            predict_answer_with_value.append(predict_answer[i])
            if gt_answer[i] == predict_answer[i]:
                correct += 1

    gt_answer = gt_answer_with_value
    predict_answer = predict_answer_with_value

    print(total)
    print(correct / total)

    from sklearn.metrics import precision_recall_fscore_support
    precison, recall, fscore, _ = precision_recall_fscore_support(gt_answer, predict_answer, average='weighted')
    print('Precision: ', precison)
    print('Recall: ', recall)
    print('Fscore: ', fscore)

    from evaluate.utils import rouge_score, bleu_score, unique_sentence_percent, root_mean_square_error, \
        mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity

    tokens_of_processed_predict_exps = [list(jieba.cut(item, cut_all=False)) for item in predict_exp]
    tokens_of_processed_gt_exps = [list(jieba.cut(item, cut_all=False)) for item in gt_exp]
    # tokens_of_processed_predict_exps = [list(item) for item in predict_exp]
    # tokens_of_processed_gt_exps = [list(item) for item in gt_exp]

    processed_gt_exps = [' '.join(list(item)) for item in predict_exp]
    processed_predict_exps = [' '.join(list(item)) for item in gt_exp]

    BLEU1 = bleu_score(tokens_of_processed_gt_exps, tokens_of_processed_predict_exps, n_gram=1, smooth=False)
    BLEU2 = bleu_score(tokens_of_processed_gt_exps, tokens_of_processed_predict_exps, n_gram=2, smooth=False)
    BLEU4 = bleu_score(tokens_of_processed_gt_exps, tokens_of_processed_predict_exps, n_gram=4, smooth=False)
    ROUGE = rouge_score(processed_gt_exps, processed_predict_exps)

    print('BLEU-1 {:7.4f}'.format(BLEU1))
    print('BLEU-2 {:7.4f}'.format(BLEU2))
    print('BLEU-4 {:7.4f}'.format(BLEU4))
    for (k, v) in ROUGE.items():
        print('{} {:7.4f}'.format(k, v))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default="../../data/test_with_annotations.csv")
    parser.add_argument("--savepath", type=str, default="../exp/test_with_chatglm.csv")
    parser.add_argument("--modelpath", type=str, default="THUDM/chatglm-6b")
    parser.add_argument("--tokenizerpath", type=str, default="THUDM/chatglm-6b")
    args = parser.parse_args()
    prediction(args)
    evaluation(args)