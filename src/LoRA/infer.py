import os
import sys
import json
import fire
import torch
import pandas as pd
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from utils.prompter import Prompter
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

class InferenceEngine:

    def __init__(self):
        self.device = device

    def load_instruction(self, instruct_dir):
        input_data = []
        with open(instruct_dir, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                d = json.loads(line)
                input_data.append(d)
        return input_data

    def load_instruction_from_csv(self, instruct_dir, prompt_idx='all'):
        input_data = []
        df = pd.read_csv(instruct_dir, dtype='str')
        if prompt_idx!='all':
            df = df[df['prompt_idx'] == str(prompt_idx)]
        dict_from_df = df.to_dict(orient='index')
        for key,value in dict_from_df.items():
            data = {}
            data['output'] = value['completion'].strip()
            data['instruction'] = value['prompt'].strip()
            input_data.append(data)
        return input_data, df

    def evaluate(self,
                 batch,
                 input=None,
                       **kwargs,
                       ):
        prompts = [self.prompter.generate_prompt(data["instruction"], input) for data in batch]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        generation_config = GenerationConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            num_beams=self.num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = self.model.generate(
                **inputs,
                generation_config=generation_config,
                # return_dict_in_generate=True,
                # output_scores=True,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=self.num_return_sequences,
            )
        outputs = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        return [self.prompter.get_response(output) for output in outputs]

    def infer_from_csv(self, instruct_dir, output_dir, prompt_id):
        input_data, df_ori = self.load_instruction_from_csv(instruct_dir, prompt_id)
        df_ori.reset_index(drop=True, inplace=True)
        col_name = 'model_result'
        batched_data = [input_data[i:i+self.batch_size] for i in range(0, len(input_data), self.batch_size)]
        model_output_dict = {col_name:[]}
        for batch in tqdm(batched_data):
            instructions = [data["instruction"] for data in batch]
            outputs = self.evaluate(batch)
            for i, output in enumerate(outputs):
                instruction = instructions[i]
                golden_output = batch[i]["output"]
                print("###infering###")
                print("###instruction###")
                print(instruction)
                print("###golden output###")
                print(golden_output)
                print("###model output###")
                print(output)
                model_output_dict[col_name].append(output)
        new_df = pd.DataFrame(model_output_dict)
        merged_df = pd.concat([df_ori, new_df], axis=1)
        merged_df.to_csv(output_dir + self.output_file_name, index=False)

    def run(self,
            load_8bit=False,
            base_model="medalpaca/medalpaca-7b",
            instruct_dir="../../data/test_prompt.csv",
            prompt_id="4",
            output_dir="output/",
            output_file_name="output.csv",
            use_lora=False,
            lora_weights="tloen/alpaca-lora-7b",
            prompt_template="med_template",
            batch_size=4,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=32,
            num_return_sequences=1
            ):
        self.output_file_name = output_file_name
        self.prompter = Prompter(prompt_template)
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model, padding_side="left")
        self.model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.num_return_sequences = num_return_sequences

        if use_lora:
            print(f"using lora {lora_weights}")
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_weights,
                torch_dtype=torch.float16,
            )
        # unwind broken decapoda-research config
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        if not load_8bit:
            self.model.half()  # seems to fix bugs for some users.

        self.model.eval()

        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        if instruct_dir != "":
            filename, file_extension = os.path.splitext(instruct_dir)
            file_extension_without_dot = file_extension[1:]
            if file_extension_without_dot == 'json':
                self.infer_from_json(instruct_dir)
            elif file_extension_without_dot == 'csv':
                self.infer_from_csv(instruct_dir, output_dir, prompt_id)
            else:
                raise ValueError
        else:
            for instruction in [
                "我感冒了，怎么治疗",
                "一个患有肝衰竭综合征的病人，除了常见的临床表现外，还有哪些特殊的体征？",
                "急性阑尾炎和缺血性心脏病的多发群体有何不同？",
                "小李最近出现了心动过速的症状，伴有轻度胸痛。体检发现P-R间期延长，伴有T波低平和ST段异常",
            ]:
                print("Instruction:", instruction)
                print("Response:", self.evaluate(instruction))
                print()

if __name__ == "__main__":
    fire.Fire(InferenceEngine().run)
