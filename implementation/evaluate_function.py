import numpy as np
import math
from implementation import sample_iterator 
# from cluster import Cluster

'''
    sequences: 存储每个序列的初始评分,访问次数,序列长度的list[dict[str, Any]]
    s_min: 初始评分的最小值
    s_max: 初始评分的最大值
    baseline: 得分最高序列的序列长度,可选项,若不使用,baseline请填-1
    c_1: (1 - math.exp(c_1 * (t_init - 1)))的超参数c_1,为100区分度较明显
'''
evaluate_function_file = 'evaluate_function.pkl'
evaluate_function_list = []
import pickle
def calculate_score(score_list: list, visit_list: list, length_list: list):
    evaluate_function_list.append((score_list, visit_list, length_list))

    with open(evaluate_function_file, 'wb') as f:
        pickle.dump(evaluate_function_list, f)

    # 转换初始评分
    s_min = min(score_list)
    s_max = max(score_list)
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
    c_v1 = 0.1
    c_v2 = 3
    c_l1 = 0.1
    c_l2 = 3
    len_baseline = min([length_list[i] for i, score in enumerate(score_list) if score == s_max])
    c_1 = 100
    temperature = 0.1
    # 计算每个序列的权重
    weights = []
    for score, t_init, visit, length in zip(score_list, transformed_initials, visit_list, length_list):
        # 测试了对于小于长度baseline的序列不加分也不减分的情况，区分度不如都减分的情况
        intensity = 1 - math.exp(c_1 * (t_init - 1))
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
    
    # 归一化权重为总和1的评分
    '''
    total_temp = sum(weights)
    weights_temp = [w - total_temp for w in weights]
    total = sum(weights_temp)
    scores = [w / total for w in weights_temp]
    index = np.argmax(scores)
    scores[index] = 1 - sum(scores[0:index]) - sum(scores[index+1:])
    '''
    '''
    weights_temp = []
    # weight < -10000 的序列不做归一化,我们认为 < -10000已经没有参考价值
    for w in weights:
        if w > -10000:
            weights_temp.append(w)
    total_temp = sum(weights_temp)
    weights_convert = [w - total_temp for w in weights_temp]
    total_convert = sum(weights_convert)
    scores = [w / total_convert for w in weights_convert]
    '''
    '''
    scores = scipy.special.softmax(weights)
    index = np.argmax(scores)
    scores[index] = 1 - sum(scores[0:index]) - sum(scores[index+1:])
    '''
    # p: 归一化函数的指数部分
    # p = 3
    # total_temp = sum(weights)
    # weights_temp = [w - total_temp for w in weights]
    # total = sum([w**p for w in weights_temp])
    # scores = [w**p / total for w in weights_temp]
    # index = np.argmax(scores)
    # scores[index] = 1 - sum(scores[0:index]) - sum(scores[index+1:])
    scores = sample_iterator.softmax(weights, temperature)
    return scores