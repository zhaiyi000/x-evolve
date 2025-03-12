import torch
from transformers import AutoModelForCausalLM
from tokenizer import Tokenizer
import json
from multiprocessing import Queue, Process
import time
import random


import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sh_and_log/gen_sentence.log'),
        logging.StreamHandler()
    ]
)


def exec_code(code_template, text):
    text = text.replace('\t', '    ')
    codes = code_template.replace('__placeholder__', text)

    with open('test_bin_packing.py', 'w') as f:
        f.write(codes)

    try:
        scope = {}
        exec(codes, scope)
        score = scope['result']
        return score
    except:
        return None


def write_records(text, score):
    text = text.replace('    ', '\t')
    with open('records.json', 'a') as f:
        json.dump(dict(text=text, score=score), f)
        f.write('\n')


def check_end(ids, tokenizer):
    token_id = ids[-1]
    if token_id == tokenizer.sep_token_id or \
        token_id == tokenizer.pad_token_id or \
        len(ids) == tokenizer.model_max_length:
        return True
    else:
        return False


def valid_worker(valid_model, input_ids, attention_mask):
    with torch.no_grad():
        valid_outputs = valid_model(input_ids=input_ids, attention_mask=attention_mask)
        valid_logits = valid_outputs.logits[:, -1]
        torch.sigmoid_(valid_logits)
        mask = valid_logits > 0.5
        res_list = [torch.where(row_mask)[0] for row_mask in mask]
        return res_list


def prob_worker(prob_model, input_ids, attention_mask, temperature):
    with torch.no_grad():
        prob_outputs = prob_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = prob_outputs.logits[:, -1]
        probs = torch.softmax(logits/temperature, dim=-1)
        res_list = []
        for i in range(input_ids.shape[0]):
            res_list.append(probs[i])
        return res_list


def exec_worker(exec_queue, exec_res_queue, tokenizer, code_template, text_score_dic, err_que):
    try:
        import os
        logging.info(f"exec_worker PID: {os.getpid()}")
        exec_res = []
        while True:
            input_ids = exec_queue.get()
            if input_ids == 'end':
                exec_res_queue.put(exec_res)
                exec_res = []
                time.sleep(0)
            else:
                text = tokenizer.decode(input_ids, add_special_tokens=False)
                if text in text_score_dic:
                    logging.info(f'repeat {text_score_dic[text]}')
                    exec_res.append(text_score_dic[text])
                else:
                    score = exec_code(code_template, text)
                    text_score_dic[text] = score
                    if score:
                        write_records(text, score)
                    else:
                        logging.info('invalid')
                    exec_res.append(score)
    except Exception as err:
        logging.info(str(err))
        import traceback
        traceback.print_exc()
        err_que.put(err)
        raise err



def get_next_token(input_ids_ori, num_samples, try_size, batch_size, tokenizer, exec_queue, exec_res_queue, valid_model, device, prob_model, temperature):
    
    input_ids_list = [input_ids_ori] * try_size
    first_token_list = []

    while len(input_ids_list) != 0:
        input_ids_batch, input_ids_list = input_ids_list[-batch_size:], input_ids_list[:-batch_size]

        max_length = max([len(x) for x in input_ids_batch])
        attention_mask = [[1] * len(x) + [0] * (max_length - len(x)) for x in input_ids_batch]
        input_ids = [x + [tokenizer.pad_token_id] * (max_length - len(x)) for x in input_ids_batch]
        print(max_length, end='       \r')

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.int64, device=device)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.int64, device=device)
        
        valid_indices_batch = valid_worker(valid_model, input_ids_tensor, attention_mask_tensor)
        probs_batch = prob_worker(prob_model, input_ids_tensor, attention_mask_tensor, temperature)

        single_indices_mask = [idx.shape[-1] == 1 for idx in valid_indices_batch]
        single_indices_list = []
        if any(single_indices_mask):
            single_indices = torch.cat([idx for idx, mask in zip(valid_indices_batch, single_indices_mask) if mask], dim=0)
            single_indices_list = single_indices.tolist()
        single_idx = 0

        new_input_ids_list = []
        for valid_indices, probs, input_ids in zip(valid_indices_batch, probs_batch, input_ids_batch):

            if valid_indices.shape[-1] == 0:
                continue
            elif valid_indices.shape[-1] == 1:
                # sampled_token = valid_indices.item()
                sampled_token = single_indices_list[single_idx]
                single_idx += 1
                new_input_ids_list.extend([input_ids + [sampled_token]])
            else:
                probs = probs[valid_indices]
                sampled_indices = torch.multinomial(probs, num_samples=num_samples, replacement=True)
                sampled_indices = valid_indices[sampled_indices.unique()]
                sampled_indices = sampled_indices.tolist()
                new_input_ids_list.extend([input_ids + [sampled_token] for sampled_token in sampled_indices])
        
        for input_ids in new_input_ids_list:
            if check_end(input_ids, tokenizer):
                first_token_list.append(input_ids)
                exec_queue.put(input_ids)
            else:
                input_ids_list.append(input_ids)

    exec_queue.put('end')
    scores = exec_res_queue.get()

    max_score = float('-inf')
    res_first_token = []
    for score, first_token in zip(scores, first_token_list):
        if score:
            if score > max_score:
                max_score = score
                res_first_token = [first_token]
            elif score == max_score:
                res_first_token.append(first_token)
    
    if len(res_first_token) == 0:
        return None, None
    else:
        return random.choice(res_first_token), max_score



def main():

    try_size = 16
    batch_size = 16
    temperature = 1
    num_samples = 1
    device = 2
    valid_path = 'clm_mcts_valid/checkpoint-258000'
    prob_path = 'clm_mcts/checkpoint-20000'
    tokenizer = Tokenizer.from_pretrained(valid_path)


    valid_model = AutoModelForCausalLM.from_pretrained(valid_path).to(device)
    valid_model.eval()

    prob_model = AutoModelForCausalLM.from_pretrained(prob_path).to(device)
    prob_model.eval()


    text_score_dic = {}
    with open('json/body.json', 'r') as f:
        lines = f.read().strip().split('\n')
        for line in lines:
            json_line = json.loads(line)
            text_score_dic[json_line['text']] = json_line['score']
    logging.info(f'len text_score_dic {len(text_score_dic)}')

    with open('bin_packing_template.py', 'r') as f:
        code_template = f.read()

    exec_queue = Queue()
    exec_res_queue = Queue()
    err_que = Queue()

    exec_process = Process(target=exec_worker, args=(exec_queue, exec_res_queue, tokenizer, code_template, text_score_dic, err_que))
    exec_process.daemon = True
    exec_process.start()

    input_ids_cpu = [1]
    outer_max_score = float('-inf')
    outer_input_ids = []

    while True:   
        sampled_input_ids, max_score = get_next_token(input_ids_cpu, num_samples, try_size, batch_size, tokenizer, exec_queue, exec_res_queue, valid_model, device, prob_model, temperature)
        if sampled_input_ids is None:
            logging.info('no valid token')
            if len(input_ids_cpu) > 1:
                input_ids_cpu.pop()
            continue

        if max_score > outer_max_score:
            outer_max_score = max_score
            outer_input_ids = sampled_input_ids
            logging.info(f'new outer max socre {max_score} {outer_max_score}')
        elif max_score == outer_max_score:
            logging.info(f'equal outer max socre {max_score} {outer_max_score}')
            outer_input_ids = random.choice([outer_input_ids, sampled_input_ids])
        else:
            logging.info(f'old outer max socre {max_score} {outer_max_score}')

        input_ids_cpu.append(outer_input_ids[len(input_ids_cpu)])
        logging.info(str(input_ids_cpu))
        if check_end(input_ids_cpu, tokenizer):
            input_ids_cpu = [1]
            outer_max_score = float('-inf')
            outer_input_ids = []
            logging.info('finish_one')
           
            
if __name__ == '__main__':
    import os
    logging.info(f"主进程 PID: {os.getpid()}")
    main()

