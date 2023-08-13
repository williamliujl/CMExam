# CMExam

## Introduction

CMExam is a dataset sourced from the Chinese National Medical Licensing Examination. It consists of 60K+ multiple-choice questions and five additional question-wise annotations, including disease groups, clinical departments, medical disciplines, areas of competency, and question difficulty levels. Alongside the dataset, comprehensive benchmarks were conducted on representative LLMs on CMExam.

<img src="https://github.com/williamliujl/CMExam/blob/main/docs/example.png" width="860" />

## Dataset Statistics
|                            | Train         | Val           | Test          | Total         |
|----------------------------|---------------|---------------|---------------|---------------|
| Question                   | 54,497        | 6,811         | 6,811         | 68,119        |
| Vocab                      | 4,545         | 3,620         | 3,599         | 4,629         |
| Max Q tokens               | 676           | 500           | 585           | 676           |
| Max A tokens               | 5             | 5             | 5             | 5             |
| Max E tokens               | 2,999         | 2,678         | 2,680         | 2,999         |
| Avg Q tokens               | 29.78         | 30.07         | 32.63         | 30.83         |
| Avg A tokens               | 1.08          | 1.07          | 1.07          | 1.07          |
| Avg E tokens               | 186.24        | 188.95        | 201.44        | 192.21        |
| Median (Q1, Q3) Q tokens   | 17 (12, 32)   | 18 (12, 32)   | 18 (12, 37)   | 18 (12, 32)   |
| Median (Q1, Q3) A tokens   | 1 (1, 1)      | 1 (1, 1)      | 1 (1, 1)      | 1 (1, 1)      |
| Median (Q1, Q3) E tokens   | 146 (69, 246) | 143 (65, 247) | 158 (80, 263) | 146 (69, 247) |


## Deployment

To deploy this project run

### Environment Setup
```
  cd src
  pip install -r requirements.txt
```
### Data Preprocess
```
  cd preprocess
  python generate_prompt.py
```

### Ptuning
```
  cd ../ptuning
  bash train.sh
  bash prediction.sh
```


## To-Do

- [ ] add statistics scripts of CMExam

- [ ] add off-the-shelf LLMs evaluation prompts and scripts

- [x] add LLMs finetuing prompts and scripts



## Citation
Benchmarking Large Language Models on CMExam -- A Comprehensive Chinese Medical Exam Dataset
https://arxiv.org/abs/2306.03030

```
@article{liu2023benchmarking,
  title={Benchmarking Large Language Models on CMExam--A Comprehensive Chinese Medical Exam Dataset},
  author={Liu, Junling and Zhou, Peilin and Hua, Yining and Chong, Dading and Tian, Zhongyu and Liu, Andrew and Wang, Helin and You, Chenyu and Guo, Zhenhua and Zhu, Lei and others},
  journal={arXiv preprint arXiv:2306.03030},
  year={2023}
}
```
