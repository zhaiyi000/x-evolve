import glob
import natsort
import json
import re
from evaluate_function import calculate_score
import numpy as np
from tokenizer import Tokenizer
from transformers import AutoModelForCausalLM
import torch
import torch.nn.functional as F


def main():
    files = []
    files.extend(glob.glob('../zy1/funsearch_llm_api/samples/*.json'))
    files.extend(glob.glob('../zy2/funsearch_llm_api/samples/*.json'))
    files = natsort.natsorted(files)

    function_dic = {}

    for file in files:
        with open(file, 'r') as f:
            info = json.load(f)
        sample_order = info['sample_order']
        function = info['function']
        score = info['score']
        decisions = info['decisions']

        if score:
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

    prob = calculate_score(score_list, visit_list, length_list)
    indices = np.random.choice(len(function_list), size=64, replace=True, p=prob)
    prompts = []
    for idx in indices:
        prompts.append(function_list[idx])

    tokenizer = Tokenizer.from_pretrained('tokenizer')
    device = 0
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

    model = AutoModelForCausalLM.from_pretrained('output/checkpoint-5000')
    model.to(device)


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
    response = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
    response = response[:, input_ids.shape[1]:]
    response = response.tolist()
    for prompt, ids in zip(prompts, response):
        decisions = [tokenizer.id_to_token(x) for x in ids]


        print()
    



if __name__ == '__main__':
    main()
