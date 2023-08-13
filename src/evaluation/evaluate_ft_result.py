# -*- coding: utf-8 -*-

# --------------------------------------------
# @FileName: calc_metrics.py
# @Author: ljl
# @Time: 2023/5/10
# @Description:
# --------------------------------------------

import pandas as pd
import jieba

filepath = 'test_predicted.csv'

csv = pd.read_csv(filepath)

gt_exp = csv['Explanation'].values.tolist()
predict_exp = csv['explanation'].values.tolist()
# process pd.na
gt_exp = [item if not pd.isna(item) else "" for item in gt_exp]
predict_exp = [item if not pd.isna(item) else "" for item in predict_exp]

# gt_answer = csv['Answer'].values.tolist()
# predict_answer = csv['answer_prediction'].values.tolist()
# gt_answer_with_value = []
# predict_answer_with_value = []
#
# total = 0.0
# correct = 0.0
# for i in range(len(gt_answer)):
#     if not pd.isna(predict_answer[i]):
#         total+=1
#         gt_answer_with_value.append(gt_answer[i])
#         predict_answer_with_value.append(predict_answer[i])
#         if gt_answer[i] == predict_answer[i]:
#             correct+=1
#
#
# gt_answer = gt_answer_with_value
# predict_answer = predict_answer_with_value
#
# print(total)
# print(correct/total)

from sklearn.metrics import precision_recall_fscore_support
precison, recall, fscore, _ = precision_recall_fscore_support(gt_answer, predict_answer, average='weighted')
print('Precision: ', precison)
print('Recall: ', recall)
print('Fscore: ', fscore)

from src.evaluation.evaluate.utils import rouge_score, bleu_score, unique_sentence_percent, root_mean_square_error, mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity

tokens_of_processed_predict_exps = [list(jieba.cut(item,cut_all=False)) for item in predict_exp]
tokens_of_processed_gt_exps = [list(jieba.cut(item,cut_all=False)) for item in gt_exp]

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
