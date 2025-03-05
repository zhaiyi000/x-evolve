import numpy as np
import math
import scipy

# from evaluate_function import calculate_score
# (100-49*t_init-50*t_init*t_init+math.log2(t_init+1)/math.log2(2))
# 1-exp(10(x-1))  (1 - math.exp(10 * (t_init - 1)))
# ln(1 + c_1 - c_1x)/ln(1 + c_1)
# math.exp(s["found"])
# (s["len"]-baseline)*(s["len"]-baseline)
# s["len"]*s["len"]
def calculate_scores(sequences, temperature):
    # 转换初始评分
    transformed_initials = [(s["initial"] + 500) / (-207.3+500) for s in sequences]
#    for t in transformed_initials:
#        print(t)
    # 计算每个序列的权重
    weights = []
    baseline = sequences[8]["len"]
#    print(baseline)
    for s, t_init in zip(sequences, transformed_initials):
        print(s["found"], s["len"])
        # 测试了对于小于长度baseline的序列不加分也不减分的情况，区分度不如都减分的情况
        if s["len"] > baseline:
            weight = s["initial"] - 0.1*s["found"]**3*((1 - math.exp(100 * (t_init - 1)))) - 0.1*s["len"]**3*((1 - math.exp(100 * (t_init - 1))))
        else:
            weight = s["initial"] - 0.1*s["found"]**3*((1 - math.exp(100 * (t_init - 1)))) - 0.1*s["len"]**3*((1 - math.exp(100 * (t_init - 1))))
        print(weight)
        weights.append(weight)
    
    # 归一化权重为总和1的评分
    '''
    total = sum(weights)
    weights_temp = [w - total for w in weights]
    total_temp = sum(weights_temp)
    scores = [w  / total_temp for w in weights_temp]
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
    total_temp = sum(weights)
    weights_temp = [w - total_temp for w in weights]
    total = sum([w**3 for w in weights_temp])
    scores = [w**3 / total for w in weights_temp]
    index = np.argmax(scores)
    scores[index] = 1 - sum(scores[0:index]) - sum(scores[index+1:])
    return scores

np.random.seed(41)
sequences = [
    {
        "initial": -500,
        "found": np.random.randint(0,100),
        "len": np.random.randint(5, 50)  # 随机生成长度5~49
    },
    {
        "initial": -280,
        "found": np.random.randint(0,100),
        "len": np.random.randint(5, 50)  # 随机生成长度5~49
    },
    {
        "initial": -212,
        "found": np.random.randint(0,100),
        "len": np.random.randint(5, 50)  # 随机生成长度5~49
    },
    {
        "initial": -211,
        "found": np.random.randint(0,100),
        "len": np.random.randint(5, 50)  # 随机生成长度5~49
    },
    {
        "initial": -208,
        "found": np.random.randint(0,100),
        "len": np.random.randint(5, 50)  # 随机生成长度5~49
    },
    {
        "initial": -207.5,
        "found": np.random.randint(0,100),
        "len": np.random.randint(5, 50)  # 随机生成长度5~49
    },
    {
        "initial": -207.4,
        "found": np.random.randint(0,100),
        "len": np.random.randint(5, 50)  # 随机生成长度5~49
    },
    {
        "initial": -207.35,
        "found": np.random.randint(0,100),
        "len": np.random.randint(5, 50)  # 随机生成长度5~49
    },
    {
        "initial": -207.3,
        "found": np.random.randint(0,100),
        "len": np.random.randint(5, 50)  # 随机生成长度5~49
    }
]
# 初始评分分布
temperature = 19
initial_scores = calculate_scores(sequences,10)

print("评分完成")
print("初始评分总和:", sum(initial_scores))  # 应输出1.0
for s in initial_scores :
    print(s)
