import natsort
import glob
import json
import copy
import torch
import math
from typing import Dict
from implementation import sample_iterator
from implementation import evaluate_function
from implementation import evaluator
from implementation import code_manipulation
from X-evolve import Sandbox, specification
from bin_packing import bin_packing_val
from config import config_type
import os

if config_type == 'bin_packing':
    trim = f'def priority(item: float, bins: np.ndarray) -> np.ndarray:\n    \"\"\"Returns priority with which we want to add item to each bin.\n\n    Args:\n        item: Size of item to be added to the bin.\n        bins: Array of capacities for each bin.\n\n    Return:\n        Array of same size as bins with priority score of each bin.\n    \"\"\"\n'

source_path = f'/root/X-evolve/bin_packing_train_OR/funsearch_llm_api/samples/*.json'
best_val_path = f'best_train_OR.json'

inputs = {'val': bin_packing_val.datasets['val_OR']}
test_input = 'val'


def get_data():
    files = []
    files.extend(glob.glob(source_path))
    files = natsort.natsorted(files)
    
    # num_choose = 10000
    # files = np.random.choice(files,size = num_choose,replace= False)

    score_list = []
    visit_list = []
    length_list = []
    function_list = []
    max_score = -1e10
    visited_decisions: dict[str, set[tuple[str]]] = {}

    for file in files:
        with open(file, 'r') as f:
            try:
                info = json.load(f)
            except:
                continue
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

exector = Sandbox()
template = code_manipulation.text_to_program(specification)
function_to_evolve = 'priority'
function_to_run = 'evaluate'
def evaluate(code_list):
    program_list = []
    for generated_code in code_list:
        generated_code = generated_code.replace(trim,'')
        new_function, program = evaluator._sample_to_program(generated_code, template, function_to_evolve)
        with open('program.txt', 'a') as f:
            f.write(program + '\n')
        program_list.append(program)
    result_list = exector.run(program_list, function_to_run=function_to_run, function_to_evolve=function_to_evolve, inputs=inputs, test_input=test_input, timeout_seconds=60)
    return result_list, program_list

def get_top_n(lst, n):
    return sorted(lst, key=lambda x: x[0], reverse=True)[:n]

prompt_sampler_dic: Dict[str, sample_iterator.SampleIterator] = {}
def get_sampler(prompt):
    if prompt not in prompt_sampler_dic:
        sampler = sample_iterator.SampleIterator(prompt)
        prompt_sampler_dic[prompt] = sampler
    return prompt_sampler_dic[prompt]

def main():
    score_list, visit_list, length_list, function_list, _, _ = get_data()
    max_score = float('-inf')
    
    best_n = [(float('-inf'),'') for _ in range(10)]
    while True:
        indices, _ = evaluate_function.calculate_score(score_list, length_list, size=128, replace=True)
        prompts = [function_list[idx] for idx in indices]

        code_list = []
        for prompt, decisions_ori in prompts:
            decisions = copy.deepcopy(decisions_ori)

            decisions = tuple(decisions)

            sampler = get_sampler(prompt)
            generated_code = sampler.get_instance_by_decisions(decisions)
            code_list.append(generated_code)
        print()

        result_list, program_list = evaluate(code_list)
        
        if result_list[1] ==  False:
            this_score_list = [x[0] for x in result_list[0] if x[1]]
        else:
            raise TimeoutError('timeout')
        
        if len(this_score_list) == 0:
            print('Running error: no valid score.')
            continue
        
        if len(this_score_list) != len(program_list):
            print('Error score exist.')
            continue
        
        for score, program in zip(this_score_list, program_list):
            best_n.append((score, program))

        best_n = get_top_n(best_n, 10)
        
        with open(best_val_path,'w') as file:
            json.dump(best_n, file)

        print('this max score:', max(this_score_list), 'global score:', max_score)
        
        if max(this_score_list) > max_score:
            max_score = max(this_score_list)

        print()

if __name__ == '__main__':
    main()