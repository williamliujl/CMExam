# -*- coding: utf-8 -*-

# --------------------------------------------
# @FileName: prompt_templates.py
# @Author: ljl
# @Time: 2023/5/15
# @Description: 
# --------------------------------------------

all_task_templates = {}

template = {}
template['prompt'] = "问题: {}, \n 选项: {}"
template['completion'] = "答案: {}"
template['id'] = "1"
all_task_templates["1"] = template

template = {}
template['prompt'] = "问题: {}, \n 选项: {}"
template['completion'] = "答案: {}"
template['id'] = "2"
all_task_templates["2"] = template

template = {}
template['prompt'] = "问题: {}, \n 选项: {}"
template['completion'] = "解释: {}"
template['id'] = "3"
all_task_templates["3"] = template

template = {}
template['prompt'] = "问题: {}, \n 选项: {}"
template['completion'] = "答案: {}. \n 解释:{}"
template['id'] = "4"
all_task_templates["4"] = template

template = {}
template['prompt'] = "问题: {}"
template['completion'] = "答案: {}"
template['id'] = "5"
all_task_templates["5"] = template

template = {}
template['prompt'] = "问题: {}"
template['completion'] = "答案: {}. \n 解释: {}"
template['id'] = "6"
all_task_templates["6"] = template