# -*- coding: utf-8 -*-

# --------------------------------------------
# @FileName: generate_prompt.py
# @Author: ljl
# @Time: 2023/5/15
# @Description: 
# --------------------------------------------

import pandas as pd
import json
import copy
import argparse
from prompt_templates import all_task_templates


def main(args):

    filepath = args.filepath
    savepath = filepath.replace(".csv", ".json")


    csv = pd.read_csv(filepath)

    # prompt_templates = ["1","2","3","4","5","6"]
    prompt_templates = args.templates.split(",")

    prompts = []

    for i,data in enumerate(csv.values):

        question = data[csv.columns.values.tolist().index("Question")]
        options = data[csv.columns.values.tolist().index("Options")]
        explanation = data[csv.columns.values.tolist().index("Explanation")]
        option_lists = options.split("\n")
        answer = data[csv.columns.values.tolist().index("Answer")]
        if pd.isna(answer):
            continue
        answer_content = ""
        for option in option_lists:
            if option.split(" ")[0] == answer:
                answer_content = option.split(" ")[-1]

        for prompt_idx in prompt_templates:
            prompt_template = copy.deepcopy(all_task_templates[prompt_idx])
            try:
                if prompt_idx == "1":
                    prompt_template["prompt"] = prompt_template["prompt"].format(question, options)
                    prompt_template["completion"] = prompt_template["completion"].format(answer)
                    prompts.append(prompt_template)
                elif prompt_idx == "2":
                    prompt_template["prompt"] = prompt_template["prompt"].format(question, options)
                    prompt_template["completion"] = prompt_template["completion"].format(answer+" "+ answer_content)
                    prompts.append(prompt_template)
                elif prompt_idx == "3":
                    prompt_template["prompt"] = prompt_template["prompt"].format(question, options)
                    prompt_template["completion"] = prompt_template["completion"].format(explanation)
                    prompts.append(prompt_template)
                elif prompt_idx == "4":
                    prompt_template["prompt"] = prompt_template["prompt"].format(question, options)
                    prompt_template["completion"] = prompt_template["completion"].format(answer+" "+ answer_content, explanation)
                    prompts.append(prompt_template)
                elif prompt_idx == "5":
                    prompt_template["prompt"] = prompt_template["prompt"].format(question)
                    prompt_template["completion"] = prompt_template["completion"].format(answer_content)
                    prompts.append(prompt_template)
                elif prompt_idx == "6":
                    prompt_template["prompt"] = prompt_template["prompt"].format(question)
                    prompt_template["completion"] = prompt_template["completion"].format(answer_content, explanation)
                    prompts.append(prompt_template)
            except Exception as e:
                print(data)

    # save json
    with open(savepath, 'w') as f:
        for prompt in prompts:
            json_file = {
                "prompt":prompt["prompt"],
                "completion":prompt["completion"],
                "id":prompt["id"]
            }
            json_str = json.dumps(json_file,ensure_ascii=False)
            f.write(json_str + '\n')
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, required=True)
    parser.add_argument("--templates", type=str, default="1,2", help="To generate prompts using different templates")
    args = parser.parse_args()
    main(args)
