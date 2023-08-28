# -*- coding: utf-8 -*-

# --------------------------------------------
# @FileName: translate.py
# @Author: ljl
# @Time: 2023/5/4
# @Description: 
# --------------------------------------------

import openai
import argparse
import os
import time
import jieba
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm
os.environ["HTTP_PROXY"] = "socks5h://127.0.0.1:13659"
os.environ["HTTPS_PROXY"] = "socks5h://127.0.0.1:13659"
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "https://127.0.0.1:7890"

def call_api(data,question_nums, model):
    results = []
    try:
        for i, content in tqdm(enumerate(data)):
            result = ""
            try:
                completion = openai.ChatCompletion.create(
                    model=model,
                    # model="gpt-4",
                    # model="gpt-4-0314",
                    messages=[{"role": "user", "content": content}]
                )
                result = completion.choices[0].message.content
            except Exception as e:
                print(str(e), flush=True)
            results.append(result)
    except Exception as e:
        print(str(e), flush=True)
        results.extend(["[]" for _ in range(len(data)-len(results))])
    return results,question_nums

def prediction(args):
    openai.api_key = args.api_key

    csv = pd.read_csv(args.filepath)
    questions = csv['Question'].values.tolist()
    options = csv['Options'].values.tolist()

    template = "返回格式为一个python列表，包含每道题的答案英文选项和解释 \n" \
               "假设你是一位医疗行业专家，请回答下列几个问题。\n" \
               "题目信息为：{} \n" \
               "注意，每个题目的回答以一个字符串保存，返回答案的英文选项，并进行简要的解释。字符串输出格式限制为“答案：**,解释：**”"
    data = []
    question_nums = []
    step = 5

    for i in range(0,len(questions),step):
        question_group = ""
        question_num = min(step, len(questions)-i)
        for j in range(question_num):
            question_group+="{}.题目信息为 {}:{}\n".format(str(j+1),questions[i+j], options[i+j].replace('\n',','))

        data.append(template.format(question_group))
        question_nums.append(question_num)

    # data = data[:2]
    # question_nums = question_nums[:2]

    # multiprocessing
    num_of_processes = 1
    pool = Pool(processes=num_of_processes)
    pool_results = []
    each_size = len(data) // num_of_processes
    for i in range(num_of_processes):
        if i<num_of_processes-1:
            pool_results.append(pool.apply_async(call_api,(data[i*each_size:(i+1)*each_size],question_nums[i*each_size:(i+1)*each_size], args.model)))
        else:
            pool_results.append(pool.apply_async(call_api, (data[i * each_size:],question_nums[i * each_size:],args.model)))
    pool.close()
    pool.join()

    import re
    results = []
    option_results = []
    explain_results = []
    for res in pool_results:
        merge_res, merge_question_num = res.get()
        final_res = []
        option_pool = []
        explain_pool = []
        for i in range(len(merge_question_num)):
            try:
                lis = merge_res[i].split("\n")
                assert len(lis) == merge_question_num[i]
                final_res.extend(lis)
                for single in list:
                    try:
                        option = re.findall(r"答案：(.*)，", single)[0]
                        exp = re.findall(r"解释：(.*)", single)[0]
                        option_pool.append(option)
                        explain_pool.append(exp)
                    except Exception as e:
                        print(single, flush=True)
                        option_pool.append("")
                        explain_pool.append("")
            except:
                print(merge_res[i],flush=True)
                final_res.extend(["" for _ in range(merge_question_num[i])])
                option_pool.extend(["" for _ in range(merge_question_num[i])])
                explain_pool.extend(["" for _ in range(merge_question_num[i])])

        results.extend(final_res)
        option_results.extend(option_pool)
        explain_results.extend(explain_pool)

    results.extend(["" for _ in range(len(csv['Question']) - len(results))])
    option_results.extend(["" for _ in range(len(csv['Question']) - len(option_results))])
    explain_results.extend(["" for _ in range(len(csv['Question']) - len(explain_results))])
    csv['raw_prediction'] = results
    csv['predicted_answer'] = option_results
    csv['predicted_explanation'] = explain_results
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
            total+=1
            gt_answer_with_value.append(gt_answer[i])
            predict_answer_with_value.append(predict_answer[i])
            if gt_answer[i] == predict_answer[i]:
                correct+=1

    gt_answer = gt_answer_with_value
    predict_answer = predict_answer_with_value

    print(total)
    print(correct/total)

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
    parser.add_argument("--savepath", type=str, default="../exp/test_with_gpt.csv")
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4")
    args = parser.parse_args()
    prediction(args)
    evaluation(args)