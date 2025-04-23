import glob
import natsort
import json
import re
import numpy as np
from tokenizer import Tokenizer
from transformers import AutoModelForCausalLM
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from funsearch_bin_packing_llm_api import Sandbox, specification
from implementation import bin_packing_utils
from typing import Dict
from implementation import code_manipulation
from implementation import sample_iterator
from implementation import evaluate_function
from implementation import evaluator
import math
import copy

tokenizer_path = f'tokenizer_512_short_1'
model_path = f'output_512_short_1/checkpoint-100000'

def get_model():
    tokenizer = Tokenizer.from_pretrained(tokenizer_path)
    device = 1

    # gen_kwargs = {
    #     "min_length": -1,
    #     "max_length": 1e4,
    #     "top_k": 0,
    #     "top_p": 1,
    #     "num_return_sequences": 1,
    #     "do_sample": True,
    #     "pad_token_id": tokenizer.pad_token_id,
    #     "eos_token_id": tokenizer.sep_token_id,
    #     # "temperature": 0,
    # }

    model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    return tokenizer, model, device



def get_data():
    files = []
    files.extend(glob.glob('/root/funsearch/log_loop1_model3_21/funsearch_llm_api/samples/*.json'))
    files = natsort.natsorted(files)

    score_list = []
    visit_list = []
    length_list = []
    function_list = []
    max_score = -1e10
    visited_decisions: dict[str, set[tuple[str]]] = {}

    for file in files:
        with open(file, 'r') as f:
            info = json.load(f)
        sample_order = info['sample_order']
        function = info['function']
        score = info['score']
        decisions = info['decisions']

        if function not in visited_decisions:
            visited_decisions[function] = set()
        visited_decisions[function].add(tuple(decisions))

        if score:
            max_score = max(max_score, score)

            score_list.append(score)
            visit_list.append(0)
            length_list.append(len(function))
            function_list.append((function, decisions))

    print('len function_dic', len(function_list))
    
    return score_list, visit_list, length_list, function_list, max_score, visited_decisions



def pad_tensor(tensors, pad_value, device):
    tokenizer = Tokenizer.from_pretrained(tokenizer_path)
    max_len = tokenizer.model_max_length
    tensors = [F.pad(seq, (0, max_len - len(seq)), value=pad_value) for seq in tensors]
    tensors = torch.stack(tensors)
    tensors = tensors.to(device)
    return tensors


exector = Sandbox()
template = code_manipulation.text_to_program(specification)
function_to_evolve = 'priority'
function_to_run = 'evaluate'
bin_packing_or3 = {'OR3': bin_packing_utils.datasets['OR3']}
def evaluate(code_list):
    program_list = []
    for generated_code in code_list:
        new_function, program = evaluator._sample_to_program(generated_code, template, function_to_evolve)
        with open('program.txt', 'a') as f:
            f.write(program + '\n')
        program_list.append(program)
    result_list = exector.run(program_list, function_to_run=function_to_run, function_to_evolve=function_to_evolve, inputs=bin_packing_or3, test_input='OR3', timeout_seconds=30)
    return result_list



prompt_sampler_dic: Dict[str, sample_iterator.SampleIterator] = {}
def get_sampler(prompt):
    if prompt not in prompt_sampler_dic:
        sampler = sample_iterator.SampleIterator(prompt)
        prompt_sampler_dic[prompt] = sampler
    return prompt_sampler_dic[prompt]


def generate(model, input_ids, attention_mask, tokenizer):
    temperature = 1
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        lm_logits = outputs.logits / temperature
        mask = input_ids == tokenizer.mask_token_id
        labels_logits = lm_logits[mask]
        probs = torch.softmax(labels_logits, dim=-1)
        indices = torch.multinomial(probs, num_samples=1)
        indices = indices.tolist()
    return indices



def main():
    tokenizer, model, device = get_model()
    model.eval()
    score_list, visit_list, length_list, function_list, max_score, visited_decisions = get_data()

    while True:
        prob = evaluate_function.calculate_score(score_list, visit_list, length_list)
        indices = np.random.choice(len(function_list), size=64, replace=True, p=prob)
        prompts = [function_list[idx] for idx in indices]
        features = tokenizer(prompts)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_labels_ids = []

        for input_ids, labels in features:
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)
            labels_ids = labels.clone()
            
            num_elements = len(labels_ids)
            replace_num = math.ceil(num_elements * 0.15)

            # with torch.no_grad():
            #     input_ids = torch.cat([input_ids, labels_ids])
            #     attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            #     max_len = tokenizer.model_max_length
            #     input_ids = F.pad(input_ids, (0, max_len - len(input_ids)), value = 0)
            #     input_ids = input_ids.to(device)
            #     attention_mask = F.pad(attention_mask, (0, max_len - len(attention_mask)), value = 0)
            #     attention_mask = attention_mask.to(device)
            #     print(input_ids.shape)
            #     # 需要把input_idspad成1920，参考84行
            #     # 打印input_ids.shape看看  64*1920
            #     outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            #     logits = outputs.logits
            #     # 打印logits.shape看看  64*21*1880
            #     print(logits.shape)
            #     probs = torch.softmax(logits, dim=-1)
            #     # topprobs = torch.topk(probs, 1, dim=-1) 
            #     # 分两步来做
            #     # 1. 直接取最小的15%
            #     _, idx= torch.sort(probs)
            #     replace_indices = idx[:replace_num]
            #     # 2. 按概率采样15%
            #     # replace_indices = torch.multinomial(sorted_probs, num_samples=replace_num)


            if replace_num > 0:
                replace_indices = torch.randperm(num_elements)[:replace_num]
                labels_ids[replace_indices] = tokenizer.mask_token_id
            
            input_ids = torch.cat([input_ids, labels_ids])
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_labels_ids.append(labels_ids)
            batch_attention_mask.append(torch.ones_like(input_ids, dtype=torch.long))
        
        batch_input_ids = pad_tensor(batch_input_ids, pad_value=0, device=device)
        batch_attention_mask = pad_tensor(batch_attention_mask, pad_value=0, device=device)
        # batch_labels = pad_tensor(batch_labels, pad_value=-100, device=device)
        # batch_labels_ids_pad = pad_tensor(batch_labels_ids, pad_value=-100, device=device)

        indices_gen = generate(model, batch_input_ids, batch_attention_mask, tokenizer)
        gen_i = 0

        code_list = []
        for (prompt, decisions_ori), ids_mask, ids_ori in zip(prompts, batch_labels_ids, batch_labels):
            decisions = copy.deepcopy(decisions_ori)
            assert len(ids_mask) == len(ids_ori)
            for idx, (id_mask, id_ori) in enumerate(zip(ids_mask, ids_ori)):
                if id_mask == tokenizer.mask_token_id:
                    gen_ids = indices_gen[gen_i]
                    gen_i += 1
                    token = tokenizer.id_to_token(gen_ids[0])
                    decisions[idx] = token

            sampler = get_sampler(prompt)
            decisions = tuple(decisions)
            if decisions in visited_decisions[prompt]:
                print('r', end='')
                continue
            visited_decisions[prompt].add(decisions)
            generated_code = sampler.get_instance_by_decisions(decisions)
            code_list.append(generated_code)
        assert gen_i == len(indices_gen)
        print()

        result_list = evaluate(code_list)
        with open('code.txt', 'a') as f:
            for code in code_list:
                f.write(code + '\n')
        print(result_list)
        if result_list[1] ==  False:
            this_score_list = [x[0] for x in result_list[0] if x[1]]
        else:
            raise TimeoutError('timeout')
        print('this max score:', max(this_score_list), 'global score:', max_score)
        if max(this_score_list) > max_score:
            max_score = max(this_score_list)

        print()




    



if __name__ == '__main__':
    main()
