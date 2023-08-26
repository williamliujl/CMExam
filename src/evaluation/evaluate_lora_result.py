# -*- coding: utf-8 -*-
# @Time    : 2023/8/26 21:28
# @Author  : Peilin Zhou
# @FileName: evaluate_lora_result.py
# @Software: PyCharm
# @E-mail  : zhoupl@pku.edu.cn
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import warnings
import fire
from evaluate.utils import rouge_score, bleu_score
warnings.filterwarnings("ignore")

def parse_options(row):
    if not isinstance(row['model_result'],str):
        return "无答案"
    if '解释' not in row['model_result']:
        original_result = row['model_result']
    else:
        original_result = row['model_result'].split('\n')[0].strip()

    options = row['选项'].split('\n')
    option_dic = {}
    for op in options:
        op_id = op.split()[0]
        op_content = op.split()[1]
        option_dic[op_id] = op_content
    predict_ops = []

    for opi,opc in option_dic.items():
        if opi in original_result or opc in original_result:
            predict_ops.append(opi)
    if len(predict_ops)>0:
        return "".join(predict_ops)
    else:
        return "无答案"
def parse_explanations(row):
    # 从'model_results'中提取答案部分（即选项）
    if not isinstance(row['model_result'],str):
        return "无答案"
    if '解释:' not in row['model_result']:
        original_result = row['model_result']
    else:
        original_result = row['model_result'].split('解释:')[1].strip()
    return original_result
def evaluate_reasoning(df):
    def add_spaces(l):
        return [' '.join(list(_)) for _ in l]
    source = '答案解析'
    target = 'parsed_explanation'
    df.dropna(subset=[source, target], inplace=True)
    tokens_predict = df[target].to_list()
    tokens_test = df[source].to_list()

    tokens_predict = add_spaces(tokens_predict)
    tokens_test = add_spaces(tokens_test)

    new_tokens_predict = [l.split() for l in tokens_predict]
    new_tokens_test = [ll.split() for ll in tokens_test]
    BLEU1 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=1, smooth=False)
    BLEU4 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=4, smooth=False)
    ROUGE = rouge_score(tokens_test, tokens_predict)

    print('BLEU-1 {:7.4f}'.format(BLEU1))
    print('BLEU-4 {:7.4f}'.format(BLEU4))
    for (k, v) in ROUGE.items():
        if 'f_score' in k:
            print('{} {:7.4f}'.format(k, v))

def evaluate_prediction(df):
    correct = df[df['parsed_option']==df['答案']].shape[0]
    total = df.shape[0]
    num_no_answer = df[df['parsed_option']=='无答案'].shape[0]

    processed_gts = df['答案'].to_list()
    processed_results = df['parsed_option'].to_list()
    precison, recall, fscore, _ = precision_recall_fscore_support(processed_gts, processed_results, average='weighted')
    print('Precision: ', precison)
    print('Recall: ', recall)
    print('Fscore: ', fscore)
    print('Acc:{}'.format(correct/total*100))
    print('The number of "No answers:"',num_no_answer)

def main(
    csv_file_path: str = "../LoRA/output/medalpaca_4.csv",
):
    df = pd.read_csv(csv_file_path)

    df['parsed_option'] = df.apply(parse_options,axis=1)
    df['parsed_explanation'] = df.apply(parse_explanations,axis=1)

    print('Evaluation of prediction:')
    evaluate_prediction(df)
    print('*'*20)
    print('Evaluation of reasoning:')
    evaluate_reasoning(df)

if __name__ == "__main__":
    fire.Fire(main)