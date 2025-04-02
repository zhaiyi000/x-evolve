import json
import os
import glob
import math
import shutil
import random
import natsort
from dataclasses import dataclass



@dataclass
class Node:
    data: dict
    source_i: int
    file_idx: int



source_dirs = ['log_loop2_model1', 'log_loop2_model2', 'log_loop2_model3']
target_dirs = ['log_loop3_model1_test']


data_list_list = []
for source_i, source_dir in enumerate(source_dirs):
    node_dir = os.path.join(source_dir, 'node')
    node_files = glob.glob(os.path.join(node_dir, '*.json'))
    node_files = natsort.natsorted(node_files)
    print(source_dir, 'have node file', len(node_files))

    this_data_list = []
    for node_file in node_files:
        file_idx = int(os.path.splitext(os.path.basename(node_file))[0])
        with open(node_file, 'r') as f:
            data = json.load(f)
        this_data_list.append(Node(data=data, source_i=source_i, file_idx=file_idx))
    data_list_list.append(this_data_list)


data_list = []
data_i_list = [0] * len(data_list_list)
progrom_dic = {}
repeat_cnt = 0
while True:
    keys = list(range(len(data_list_list)))
    random.shuffle(keys)

    all_end = True
    for key in keys:
        data_i = data_i_list[key]
        this_data_list = data_list_list[key]
        if data_i < len(this_data_list):
            all_end = False
            data = this_data_list[data_i]
            data_i_list[key] += 1

            progrom = data.data['program']['body']
            if progrom in progrom_dic:
                repeat_cnt += 1
                print(data.file_idx, end=' ')
                assert data.data['score'] == progrom_dic[progrom]
                continue
            progrom_dic[progrom] = data.data['score']
            data_list.append(data)
    
    if all_end:
        break


# random.shuffle(data_list)
data_list.sort(key=lambda x: -x.data['score'])


data_dic_list: list[dict[int, list[Node]]] = [{}]
last_data = data_list[0]
if last_data.source_i not in data_dic_list[-1]:
    data_dic_list[-1][last_data.source_i] = []
data_dic_list[-1][last_data.source_i].append(last_data)
for this_data in data_list[1:]:
    if last_data.data['score'] != this_data.data['score']:
        data_dic_list.append({})
    if this_data.source_i not in data_dic_list[-1]:
        data_dic_list[-1][this_data.source_i] = []
    data_dic_list[-1][this_data.source_i].append(this_data)
    last_data = this_data
assert len(data_list) == sum(len(x) for dic in data_dic_list for x in dic.values())


total_cnt = len(data_list)
keep_cnt = math.ceil(total_cnt * 0.15)
one_percent = math.ceil(total_cnt * 0.01)
print('total_cnt', total_cnt)
print('repeat_cnt', repeat_cnt)
print('keep_cnt', keep_cnt)
print('one_percent', one_percent)


select_list: list[Node] = []
for data_dic in data_dic_list:
    for source_i, data_list in data_dic.items():
        print(data_list[0].data['score'], 'source_i', source_i, 'cnt', len(data_list))
    #     score_list = [node.data['score'] for node in data_list]
    #     source_list = [node.source_i for node in data_list]
    #     assert max(score_list) == min(score_list)
    #     assert max(source_list) == min(source_list)
    #     data_list.sort(key=lambda x: x.file_idx)

    data_i_list = {key: 0 for key in data_dic.keys()}

    one_i = 0
    while True:
        keys = list(data_dic.keys())
        # random.shuffle(keys)
        all_end = True
        for key in keys:
            data_list = data_dic[key]
            data_i = data_i_list[key]
            if data_i < len(data_list):
                select_list.append(data_list[data_i])
                data_i_list[key] += 1
                one_i += 1
                all_end = False
                if len(select_list) == keep_cnt or one_i == one_percent:
                    break
        if len(select_list) == keep_cnt or one_i == one_percent or all_end:
            break
    if len(select_list) == keep_cnt:
        break

assert len(select_list) == keep_cnt

for target_dir in target_dirs:

    print('keep node file', len(select_list))
    node_dir = os.path.join(target_dir, 'node')


    if os.path.exists(node_dir):
        y_or_n = 'y'  # input('delete node dir? [n]')
        if y_or_n == 'y':
            print('delete node dir!')
            shutil.rmtree(node_dir)


    os.makedirs(node_dir, exist_ok=True)
    for data_i, node in enumerate(select_list):
        print(node.data['score'], node.source_i, node.file_idx)
        node_file = os.path.join(node_dir, f'{data_i}.json')
        with open(node_file, 'w') as f:
            json.dump(node.data, f)


