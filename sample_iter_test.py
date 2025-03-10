import pickle
import numpy as np

instance_i = 2
while True:
    
    sample_iter_file = f'sample_iter/sample_iter/{instance_i}.pkl'
    with open(sample_iter_file, 'rb') as f:
        score_list = pickle.load(f)
        print(score_list)
    input()
    instance_i += 1