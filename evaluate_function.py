import numpy as np
import math
from cluster import Cluster

'''
    sequences: 存储每个序列的初始评分,访问次数,序列长度的list[dict[str, Any]]
    s_min: 初始评分的最小值
    s_max: 初始评分的最大值
    baseline: 得分最高序列的序列长度,可选项,若不使用,baseline请填-1
    c_1: (1 - math.exp(c_1 * (t_init - 1)))的超参数c_1,为100区分度较明显
'''
def calculate_score(sequences: list[Cluster], s_min: float, s_max: float, baseline: int, c_1: float | int):
    # 转换初始评分
    transformed_initials = [(s._score - s_min) / (s_max - s_min) for s in sequences]
    '''
    C_v1: 超参数C_v1,f(v)的系数部分,初始为0.1
    C_v2: 超参数C_v2,f(v)的指数部分,初始为3
    C_l1: 超参数C_l1,f(l)的系数部分,初始为0.1
    C_l2: 超参数C_l2,f(l)的指数部分,初始为3
    '''
    C_v1 = 0.1, C_v2 = 3, C_l1 = 0.1, C_l2 = 3
    # 计算每个序列的权重
    weights = []
    for s, t_init in zip(sequences, transformed_initials):
        # 测试了对于小于长度baseline的序列不加分也不减分的情况，区分度不如都减分的情况
        if baseline > 0:
            if s._min_length > baseline:
                weight = s._score - C_v1*s._visit**C_v2*(1 - math.exp(c_1 * (t_init - 1))) - C_l1*(s._min_length-baseline)**C_l2*(1 - math.exp(c_1 * (t_init - 1)))
            else:
                weight = s._score - C_v1*s._visit**C_v2*(1 - math.exp(c_1 * (t_init - 1)))
        else:
            weight = s._score - C_v1*s._visit**C_v2*(1 - math.exp(c_1 * (t_init - 1))) - C_l1*s._min_length**C_l2*(1 - math.exp(c_1 * (t_init - 1)))
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
    p = 3
    total_temp = sum(weights)
    weights_temp = [w - total_temp for w in weights]
    total = sum([w**p for w in weights_temp])
    scores = [w**p / total for w in weights_temp]
    index = np.argmax(scores)
    scores[index] = 1 - sum(scores[0:index]) - sum(scores[index+1:])
    return scores