from evaluate_function import *
import json
import numpy as np


def test1():
    with open('implementation/evaluate_function_test.json', 'r') as f:
        args_list = json.load(f)
    for args in args_list:
        calculate_score(*args)

def test2():

    weights = np.array([300, 355, 385, 410, 440, 464, 496, 512], dtype=np.float32)

    scores = sample_iterator.softmax(weights, 50)

    np.set_printoptions(suppress=True)
    print(weights)
    print(scores)

def test3():
    probs = get_exp_decay_probs_fixed_first_prob(10, first_prob=0.4, tol=1e-8)
    print(probs)


    # (fun) ➜  ~/funsearch/implementation python evaluate_function_test.py
    # search_i 32
    # lambd 0.6921614293000091
    # length 10
    # probs [0.5        0.25024656 0.12524668 0.0626851  0.03137346 0.0157022
    # 0.00785884 0.0039333  0.00196859 0.00098526]
    # (fun) ➜  ~/funsearch/implementation python evaluate_function_test.py
    # search_i 30
    # lambd 0.5066306637788702
    # length 10
    # probs [0.4        0.24100891 0.14521323 0.08749421 0.05271721 0.03176329
    # 0.01913809 0.01153113 0.00694776 0.00418618]
    # (fun) ➜  ~/funsearch/implementation 


test3()