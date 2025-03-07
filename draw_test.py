import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from implementation import sample_iterator

def calculate_score(score_list: list, visit_list: list, length_list: list) -> np.array:
    s_min = min(score_list)
    s_max = max(score_list)
    if s_max == s_min:
        transformed_initials = np.ones_like(score_list)
    else:
        transformed_initials = [(s - s_min) / (s_max - s_min) for s in score_list]
    c_v1 = 0.1
    c_v2 = 3
    c_l1 = 0.1
    c_l2 = 3
    len_baseline = min([length_list[i] for i, score in enumerate(score_list) if score == s_max])
    c_1 = 10
    temperature = 0.1
    weights = []
    for score, t_init, visit, length in zip(score_list, transformed_initials, visit_list, length_list):
        intensity = 1 - math.exp(c_1 * (t_init - 1))
        if len_baseline > 0:
            if length > len_baseline:
                weight = score - c_v1 * (visit ** c_v2) * intensity - c_l1 * ((length / len_baseline) ** c_l2) * intensity
            else:
                weight = score - c_v1 * (visit ** c_v2) * intensity
        else:
            raise Exception('todo')
            weight = score - c_v1 * (visit ** c_v2) * intensity - c_l1 * ((length / len_baseline) ** c_l2) * intensity
        weights.append(weight)
    scores = sample_iterator.softmax(weights, temperature)
    return scores

evaluate_function_file = 'evaluate_function.pkl'
evaluate_function_list = []
with open(evaluate_function_file, 'rb') as f:
    evaluate_function_list = pickle.load(f)
i = 1
# for index in range(0, 174):
score_list, visit_list, length_list = evaluate_function_list[173]
i += 1 
number = str(i)
with open('test_log/test1_' + number + '.txt', 'w') as f:
    f.write(str(max(score_list)) + '\n')
    f.write(str(min(score_list)) + '\n')
    f.write(str(np.array(score_list)) + '\n')
    # f.write(str(visit_list) + '\n')
    # f.write(str(length_list) + '\n')
plt.figure(figsize=(10, 6))
x = np.array(score_list)
y = calculate_score(score_list, visit_list, length_list)
plt.scatter(x, y, marker='*',label='scores--score_list')
plt.title('test1_' + number)
with open('test_log/test1_' + number + '.txt', 'a') as f:
    f.write(str(max(y)) + '\n')
    f.write(str(min(y)) + '\n')
    f.write(str(y))
'''
plt.figure(figsize=(10, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y, label='sin(x)')
plt.title('Simple plot')
'''

plt.xlabel('score_list')
plt.ylabel('scores')
plt.legend()
plt.savefig('test_log/test1_' + number + '.png')
plt.show()
