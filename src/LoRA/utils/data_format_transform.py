# -*- coding: utf-8 -*-
# @Time    : 2023/5/17 01:01
# @Author  : Peilin Zhou
# @FileName: data_format_transform.py
# @Software: PyCharm
# @E-mail  : zhoupl@pku.edu.cn
import json
import os
import argparse

def filter_and_convert(input_file, target_id, sample=None):
    filtered_data = []
    target_id = str(target_id)

    output_file_name = os.path.splitext(input_file)[0]
    if target_id is not None:
        output_file_name += '_' + str(target_id)
    else:
        output_file_name += '_all'

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if target_id is None or target_id=='all' or data['id'] == target_id:
                filtered_data.append({
                    'instruction': data['prompt'],
                    'input': '',
                    'output': data['completion']
                })

    output_file = output_file_name + '.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        if sample:
            for data in filtered_data[:sample]:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        else:
            for data in filtered_data:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"Filtered file is saved to：{output_file}")
    return output_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter and convert JSON file.')
    parser.add_argument('input_file', type=str, help='path to the input JSON file', default='data/train_prompt.json')
    parser.add_argument('target_id', type=str, nargs='?', default=None, help='target ID for filtering (optional)')
    args = parser.parse_args()

    input_file_path = args.input_file
    target_id = args.target_id

    filter_and_convert(input_file_path, target_id)