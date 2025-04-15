import numpy as np
import math
from implementation import sample_iterator 
from config import *
from sklearn.cluster import KMeans
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
        return 1 - t_init
    elif config_type == 'cycle_graphs':
        return (1 - math.exp(c_1 * (t_init - 1))) / (1 - math.exp(c_1 * -1))
    elif config_type == 'admissible_set':
        raise Exception('wrong type')
    else:
        raise Exception('wrong type')


def get_exp_decay_probs_fixed_first_prob(length, first_prob=0.4, tol=1e-8):
    if length < 1:
        return []

    # Binary search λ to make sum(p) = 1 where p0 = first_prob
    def compute_sum(lambd):
        return sum(first_prob * np.exp(-lambd * i) for i in range(length))

    # Binary search for λ
    low, high = 0.00001, 100.0
    for search_i in range(100):
        mid = (low + high) / 2
        total = compute_sum(mid)
        if abs(total - 1.0) < tol:
            break
        if total > 1.0:
            low = mid
        else:
            high = mid

    # Final λ
    lambd = (low + high) / 2

    # Now compute actual probs
    probs = [first_prob * np.exp(-lambd * i) for i in range(length)]
    probs = np.array(probs)
    probs /= probs.sum()  # Just to clean up any float noise
    
    print('search_i', search_i)
    print('lambd', lambd)
    print('length', length)
    print('probs', probs)

    return probs

len_prob_dic = {}
def get_probs(length):
    if length not in len_prob_dic:
        probs = get_exp_decay_probs_fixed_first_prob(length=length)
        len_prob_dic[length] = probs
    return len_prob_dic[length]


def calculate_score(score_list: list, size: int, replace: bool):
    # evaluate_function_list.append((score_list, visit_list, length_list))
    # with open(evaluate_function_file, 'wb') as f:
    #     pickle.dump(evaluate_function_list, f)
    # 转换初始评分
    
    # if evaluate_function_mask_half:
    #     min_weight = min(score_list)
    #     indices = np.random.choice(len(score_list), len(score_list)//2, replace=False)
    #     for idx in indices:
    #         score_list[idx] = min_weight
    while True:

        score_dic = {}
        for score_i, score in enumerate(score_list):
            if score not in score_dic:
                score_dic[score] = []
            score_dic[score].append(score_i)

        score_list_list = list(score_dic.items())
        score_list_list.sort(key=lambda x: -x[0])
        
        if len(score_list_list) < 10:
            segment_list = [score_indices for score, score_indices in score_list_list]
        else:
            first_score, first_indices = score_list_list[0]
            remain_seg = score_list_list[1:]
            remain_score_list = [x[0] for x in remain_seg]
            assert len(remain_score_list) >= 9, 'len(remain_score_list) >= 9'
            kmeans = KMeans(n_clusters=9, random_state=0).fit(np.array(remain_score_list).reshape(-1, 1))
            assert len(remain_score_list) == len(kmeans.labels_), 'len(remain_score_list) == len(kmeans.labels_)'
            remain_score_dic = {}
            for (score, score_indices), clu_i in zip(remain_seg, kmeans.labels_):
                if clu_i not in remain_score_dic:
                    remain_score_dic[clu_i] = []
                remain_score_dic[clu_i].extend(score_indices)
            segment_list = [first_indices] + list(remain_score_dic.values())
        print_segment_list = segment_list[:3]
        for seg in print_segment_list:
            print(score_list[seg[0]], len(seg), score_list[seg[-1]])
        seg_indices = np.random.choice(len(segment_list), size=size, p=get_probs(len(segment_list)), replace=replace)
        indices = [np.random.choice(segment_list[seg_idx], size=1)[0] for seg_idx in seg_indices]
        if len(indices) == 2 and indices[0] == indices[1]:
            print('same indices', [score_list[i] for i in indices])
        else:
            for idx in indices:
                print(score_list[idx])
            break
    return indices

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