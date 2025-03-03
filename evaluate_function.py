import numpy as np
import math

'''
    sequences: 存储每个序列的初始评分,访问次数,序列长度的list[dict[str, Any]]
    s_min: 初始评分的最小值
    s_max: 初始评分的最大值
    baseline: 得分最高序列的序列长度,可选项,若不使用,baseline请填-1
    c_1: (1 - math.exp(c_1 * (t_init - 1)))的超参数c_1,为100区分度较明显
'''
def calculate_score(sequences: list[dict[str, float | int]], s_min: float, s_max: float, baseline: int, c_1: float | int):
    # 转换初始评分
    transformed_initials = [(s["initial"] - s_min) / (s_max - s_min) for s in sequences]

    # 计算每个序列的权重
    weights = []
    for s, t_init in zip(sequences, transformed_initials):
        # 测试了对于小于长度baseline的序列不加分也不减分的情况，区分度不如都减分的情况
        if baseline > 0:
            if s["len"] > baseline:
                weight = s["initial"] - 0.1*s["found"]**3*(1 - math.exp(c_1 * (t_init - 1))) - 0.1*(s["len"]-baseline)**3*(1 - math.exp(c_1 * (t_init - 1)))
            else:
                weight = s["initial"] - 0.1*s["found"]**3*(1 - math.exp(c_1 * (t_init - 1)))
        else:
            weight = s["initial"] - 0.1*s["found"]**3*(1 - math.exp(c_1 * (t_init - 1))) - 0.1*s["len"]**3*(1 - math.exp(c_1 * (t_init - 1)))
        # print(weight)
        weights.append(weight)
    
    # 归一化权重为总和1的评分
    total_temp = sum(weights)
    weights_temp = [w - total_temp for w in weights]
    total = sum(weights_temp)
    scores = [w / total for w in weights_temp]
    return scores