import numpy as np
import math
from implementation import sample_iterator 
from config import *
# from cluster import Cluster

'''
    sequences: 存储每个序列的初始评分,访问次数,序列长度的list[dict[str, Any]]
    s_min: 初始评分的最小值
    s_max: 初始评分的最大值
    baseline: 得分最高序列的序列长度,可选项,若不使用,baseline请填-1
    c_1: (1 - math.exp(c_1 * (t_init - 1)))的超参数c_1,为100区分度较明显
'''
# evaluate_function_file = 'evaluate_function.pkl'
# evaluate_function_list = []
# import pickle


def cal_intensity(c_1, t_init, s_max):
    if config_type == 'bin_packing':
        return (1 - math.exp(c_1 * (t_init - 1))) / (1 - math.exp(c_1 * -1))
    elif config_type == 'cap_set':
        return -t_init + 1
        # return (1 - math.exp(c_1 * (t_init - 1))) / (1 - math.exp(c_1 * -1))
        # return math.exp(c_1 * -t_init) - math.exp(c_1 * -1)
    elif config_type == 'cycle_graphs':
        return (1 - math.exp(c_1 * (t_init - 1))) / (1 - math.exp(c_1 * -1))
        # return -t_init + 1
    else:
        raise Exception('wrong type')


def calculate_score(score_list: list, visit_list: list, length_list: list):
    # evaluate_function_list.append((score_list, visit_list, length_list))
    # with open(evaluate_function_file, 'wb') as f:
    #     pickle.dump(evaluate_function_list, f)
    # 转换初始评分
    
    # if evaluate_function_mask_half:
    #     min_weight = min(score_list)
    #     indices = np.random.choice(len(score_list), len(score_list)//2, replace=False)
    #     for idx in indices:
    #         score_list[idx] = min_weight

    s_min = min(score_list)
    s_max = max(score_list)
    # dump.append(score_list)
    if evaluate_function_mask_half:
        mask_ratio = 0.6
        if s_max == s_min:
            transformed_initials = np.ones_like(score_list)
        else:
            indices = np.random.choice(len(score_list), math.floor(mask_ratio * len(score_list)), replace=False)
            for idx in indices:
                score_list[idx] = s_min
            s_max = max(score_list)
            # dump.append(score_list)
            if s_max == s_min:
                transformed_initials = np.ones_like(score_list)
            else:
                transformed_initials = [(s - s_min) / (s_max - s_min) for s in score_list]
    else:
        if s_max == s_min:
            transformed_initials = np.ones_like(score_list)
        else:
            transformed_initials = [(s - s_min) / (s_max - s_min) for s in score_list]
    '''
    C_v1: 超参数C_v1,f(v)的系数部分,初始为0.1
    C_v2: 超参数C_v2,f(v)的指数部分,初始为3
    C_l1: 超参数C_l1,f(l)的系数部分,初始为0.1
    C_l2: 超参数C_l2,f(l)的指数部分,初始为3
    '''
    c_v1 = evaluate_function_c_v1
    c_v2 = 3
    c_l1 = evaluate_function_c_l1
    c_l2 = 3
    len_baseline = min([length_list[i] for i, score in enumerate(score_list) if score == s_max])
    c_1 = evaluate_function_c_1
    temperature = evaluate_function_temperature
    # 计算每个序列的权重
    weights = []
    for score, t_init, visit, length in zip(score_list, transformed_initials, visit_list, length_list):
        # 测试了对于小于长度baseline的序列不加分也不减分的情况，区分度不如都减分的情况
        intensity = cal_intensity(c_1, t_init, s_max)
        if len_baseline > 0:
            if length > len_baseline:
                weight = score - c_v1 * (visit ** c_v2) * intensity - c_l1 * ((length / len_baseline) ** c_l2) * intensity
            else:
                weight = score - c_v1 * (visit ** c_v2) * intensity
        else:
            raise Exception('todo')
            weight = score - c_v1 * (visit ** c_v2) * intensity - c_l1 * ((length / len_baseline) ** c_l2) * intensity
        # print(weight)
        weights.append(weight)

    scores = sample_iterator.softmax(weights, temperature)
    return scores