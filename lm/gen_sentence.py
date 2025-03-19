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


def main():
    files = []
    files.extend(glob.glob('../zy1/funsearch_llm_api/samples/*.json'))
    files.extend(glob.glob('../zy2/funsearch_llm_api/samples/*.json'))
    files = natsort.natsorted(files)

    function_dic = {}
    max_score = -1e10

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
        "eos_token_id": tokenizer.sep_token_id
    }

    model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained('output/checkpoint-10000')
    model.to(device)


    for file in files:
        with open(file, 'r') as f:
            info = json.load(f)
        sample_order = info['sample_order']
        function = info['function']
        score = info['score']
        decisions = info['decisions']

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


    while True:
        prob = evaluate_function.calculate_score(score_list, visit_list, length_list)
        indices = np.random.choice(len(function_list), size=64, replace=True, p=prob)
        prompts = []
        for idx in indices:
            prompts.append(function_list[idx])

        tokens = tokenizer(prompts)

        def pad_tensor(idx):
            tensors = [torch.tensor(f[idx], dtype=torch.long) for f in tokens]
            max_len = max(len(seq) for seq in tensors)
            tensors = [F.pad(seq, (max_len - len(seq), 0), value=0) for seq in tensors]
            tensors = torch.stack(tensors)
            tensors = tensors.to(device)
            return tensors
        
        input_ids = pad_tensor(0)
        attention_mask = pad_tensor(1)

        response = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
        response = response[:, input_ids.shape[1]:]
        response = response.tolist()

        prompt_sampler_dic: Dict[str, sample_iterator.SampleIterator] = {}
        program_list = []
        exector = Sandbox()
        template = code_manipulation.text_to_program(specification)
        function_to_evolve = 'priority'
        function_to_run = 'evaluate'

        for prompt, ids in zip(prompts, response):
            decisions = []
            for id in ids:
                if id == tokenizer.sep_token_id:
                    break
                token = tokenizer.id_to_token(id)
                decisions.append(token)

            if prompt not in prompt_sampler_dic:
                sampler = sample_iterator.SampleIterator(prompt)
                prompt_sampler_dic[prompt] = sampler
            sampler = prompt_sampler_dic[prompt]
            generated_code = sampler.get_instance_by_decisions(decisions)

            new_function, program = evaluator._sample_to_program(generated_code, template, function_to_evolve)
            program_list.append(program)

        bin_packing_or3 = {'OR3': bin_packing_utils.datasets['OR3']}
        result_list = exector.run(program_list, function_to_run=function_to_run, function_to_evolve=function_to_evolve, inputs=bin_packing_or3, test_input='OR3', timeout_seconds=30)
        this_score_list = [x[0] for x in result_list if x[1]]
        print('this max score:', max(this_score_list), 'global score:', max_score)
        if max(this_score_list) > max_score:
            max_score = max(this_score_list)

        print()




    



if __name__ == '__main__':
    main()
