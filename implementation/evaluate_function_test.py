from evaluate_function import *
import json


if __name__ == '__main__':
    with open('implementation/evaluate_function_test.json', 'r') as f:
        args_list = json.load(f)
    for args in args_list:
        calculate_score(*args)
