import json
import numpy as np
from tokenizer import Tokenizer
from datasets import Dataset, DatasetDict


def normalize(x):
    x = np.array(x)
    length = len(x)
    if length == 0:
        raise ValueError("Input array cannot be empty.")
    x = x + 1 / length
    total = x.sum()
    if total == 0:
        raise ValueError("Sum of elements cannot be zero after adjustment.")
    return x / total


def softmax(x):
    x = np.array(x)
    e_x = np.exp(x - np.max(x))
    result = e_x / e_x.sum(axis=-1, keepdims=True)
    return result.tolist()


def make_dataset():
    tokenizer_path = 'tokenizer'
    tokenizer = Tokenizer.from_pretrained(tokenizer_path)
    with open('json/body.json', 'r') as f:
        lines = f.read().strip().split('\n')

    text_list = []
    json_list = []
    # lines = lines[:100]
    for line in lines:
        json_line = json.loads(line)
        text_list.append(json_line['text'])
        json_list.append(json_line)

    tokens = tokenizer.batch_encode(text_list)
    input_ids_list = tokens['input_ids']

    ids_reward_list_dic = {}

    for line_i, (json_line, input_ids) in enumerate(zip(json_list, input_ids_list)):
        print(f'{line_i:5}/{len(json_list):5}', end='\r')
        rewards = json_line['score']

        for i in range(0, len(input_ids), 1):
            ids_input = input_ids[:i]
            id_label = input_ids[i]
            ids_input_str = str(ids_input)
            if ids_input_str not in ids_reward_list_dic:
                ids_reward_list_dic[ids_input_str] = dict(ids=ids_input, labels={})
            if id_label not in ids_reward_list_dic[ids_input_str]['labels']:
                ids_reward_list_dic[ids_input_str]['labels'][id_label] = [rewards]
            else:
                ids_reward_list_dic[ids_input_str]['labels'][id_label].append(rewards)

    ids_reward_list_list = list(ids_reward_list_dic.values())
    ids_reward_list_list.sort(key=lambda x: len(x['ids']))

    datas = []
    temperature = 1
    for item in ids_reward_list_list:
        ids = item['ids']
        labels = item['labels']
        labels_new = {}
        for id_label, rewards in labels.items():
            visit_count = len(rewards)
            reward = np.max(rewards)
            labels_new[id_label] = dict(visit_count=visit_count, reward=reward/temperature, score=reward)

        rewards = softmax([x['reward'] for x in labels_new.values()])
        for (id_label, dic), reward in zip(labels_new.items(), rewards):
            dic['reward'] = reward

        labels_new = dict(sorted(labels_new.items(), key=lambda item: item[1]['reward'], reverse=True))
        datas.append(dict(ids=ids, labels=labels_new))

    visit_weight_dic = {}
    rewards_dic = {}
    for data_i, data in enumerate(datas):
        print(f'{data_i:5}/{len(datas):5}', end='\r')
        ids = data['ids']
        labels = data['labels']

        rewards_dic[str(ids)] = labels

        visit_count_sum = sum([x['visit_count'] for x in labels.values()])

        for token_id, info in labels.items():
            visit_count = info['visit_count']

            ids.append(token_id)
            visit_weight_dic[str(ids)] = visit_count / visit_count_sum
            ids.pop()

    with open('json/body_visit_weight.json', 'w') as f:
        json.dump(visit_weight_dic, f, indent=2)

    with open('json/body_rewards.json', 'w') as f:
        json.dump(rewards_dic, f, indent=2)


if __name__ == '__main__':
    make_dataset()