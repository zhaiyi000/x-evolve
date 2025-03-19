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
import bin_packing_utils
from typing import Dict
from implementation import code_manipulation
from implementation import sample_iterator
from implementation import evaluate_function
from implementation import evaluator



def get_model():
    tokenizer = Tokenizer.from_pretrained('tokenizer')
    device = 0

    gen_kwargs = {
        "min_length": -1,
        "max_length": 1e4,
        "top_k": 0,
        "top_p": 1,
        "num_return_sequences": 1,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.sep_token_id,
        "temperature": 0,
    }

    model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained('output/checkpoint-13000')
    model.to(device)
    return tokenizer, model, gen_kwargs, device



def get_data():
    files = []
    files.extend(glob.glob('../zy1/funsearch_llm_api/samples/*.json'))
    files.extend(glob.glob('../zy2/funsearch_llm_api/samples/*.json'))
    files = natsort.natsorted(files)

    function_dic = {}
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
            if function not in function_dic:
                function_dic[function] = score
            else:
                function_dic[function] = max(score, function_dic[function])

    print('len function_dic', len(function_dic))
    score_list = []
    visit_list = []
    length_list = []
    function_list = []
    for function, score in function_dic.items():
        score_list.append(score)
        visit_list.append(0)
        length_list.append(len(function))
        function_list.append(function)

    return score_list, visit_list, length_list, function_list, max_score, visited_decisions



def pad_tensor(idx, tokens, device):
    tensors = [torch.tensor(f[idx], dtype=torch.long) for f in tokens]
    max_len = max(len(seq) for seq in tensors)
    tensors = [F.pad(seq, (max_len - len(seq), 0), value=0) for seq in tensors]
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
        program_list.append(program)
    result_list = exector.run(program_list, function_to_run=function_to_run, function_to_evolve=function_to_evolve, inputs=bin_packing_or3, test_input='OR3', timeout_seconds=30)
    return result_list



prompt_sampler_dic: Dict[str, sample_iterator.SampleIterator] = {}
def get_sampler(prompt):
    if prompt not in prompt_sampler_dic:
        sampler = sample_iterator.SampleIterator(prompt)
        prompt_sampler_dic[prompt] = sampler
    return prompt_sampler_dic[prompt]



def main():
    tokenizer, model, gen_kwargs, device = get_model()
    score_list, visit_list, length_list, function_list, max_score, visited_decisions = get_data()

    while True:
        prob = evaluate_function.calculate_score(score_list, visit_list, length_list)
        indices = np.random.choice(len(function_list), size=64, replace=True, p=prob)
        prompts = [function_list[idx] for idx in indices]
        tokens = tokenizer(prompts)
        input_ids = pad_tensor(0, tokens, device)
        attention_mask = pad_tensor(1, tokens, device)

        response = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
        response = response[:, input_ids.shape[1]:]
        response = response.tolist()

        code_list = []

        for prompt, ids in zip(prompts, response):
            decisions = []
            for id in ids:
                if id == tokenizer.sep_token_id:
                    break
                token = tokenizer.id_to_token(id)
                decisions.append(token)

            sampler = get_sampler(prompt)
            decisions = tuple(decisions)
            if decisions in visited_decisions[prompt]:
                print('r', end='')
                continue
            visited_decisions[prompt].add(decisions)
            generated_code = sampler.get_instance_by_decisions(decisions)
            code_list.append(generated_code)
        print()

        result_list = evaluate(code_list)
        this_score_list = [x[0] for x in result_list if x[1]]
        print('this max score:', max(this_score_list), 'global score:', max_score)
        if max(this_score_list) > max_score:
            max_score = max(this_score_list)

        print()




    



if __name__ == '__main__':
    main()
