import itertools
import os
import re
# import random
import numpy as np


FILE_DEBUG_MODE = False
MIN_SCORE = -1e10


def softmax(x, temperature, min_prob=1e-6):
    x = np.array(x)
    x = x / temperature
    x -= np.max(x)
    exp_x = np.exp(x)
    probs = exp_x / np.sum(exp_x)
    
     # 强制最小概率保护
    probs = np.clip(probs, min_prob, None)
    probs = probs / np.sum(probs)
    
    return probs


class SampleIterator:
    def __init__(
        self, code: str
    ):
        self._code = code
        # self._sample_name = sample_name
        # self._store_folder_name = store_folder_name
        self._regular = "tunable\(\[(.*?)\]\)"
        self._split = ','
        self._temperature = 0.1

        matches = list(re.finditer(self._regular, self._code))
        matches_update = False
        tunable = []
        for match in reversed(matches):
            options = match.group(1).split(self._split)
            options = [x.strip() for x in options]
            if len(options) == 0:
                raise Exception('options\'s len is zero')
            elif len(options) == 1:
                start, end = match.span()
                self._code = self._code[:start] + options[0] + self._code[end:]
                matches_update = True
            else:
                tunable.append(options)
        if matches_update:
            matches = list(re.finditer(self._regular, self._code))
        tunable.reverse()

        # indices_list = itertools.product(*())
        # all_comb = [(indices, [tunable[space_i][i] for space_i, i in enumerate(indices)]) for indices in indices_list]
        # score_list = [[None] * len(space) for space in tunable]

        # count = 0
        # instances = []
        # for indices, comb in all_comb:
        #     function_code = self._code
        #     for match, item in zip(reversed(matches), reversed(comb)):
        #         start, end = match.span()
        #         function_code = function_code[:start] + item + function_code[end:]
        #     # self.save_function(function_code, count)
        #     # count += 1
        #     instances.append((indices, function_code))
        #     # if 'tunable(' in function_code:
        #     #     raise Exception('tuneable in function_code')
        # self.instances = instances
        self.score_list = [[MIN_SCORE] * len(space) for space in tunable]
        self.score_record = [[[MIN_SCORE]]* len(space)  for space in tunable]
        self.visited_indices = set()
        self.tunable = tunable
        self.matches = matches


    def calculate_probability(self):
        probability = []
        for scores in self.score_list:
            max_score = max(scores)
            scores = [x if x != MIN_SCORE else max_score for x in scores]
            prob = softmax(scores, self._temperature)
            probability.append(prob)
        return probability
    

    def get_instance(self, indices):
        function_code = self._code
        for match, space, idx in zip(reversed(self.matches), reversed(self.tunable), reversed(indices)):
            start, end = match.span()
            function_code = function_code[:start] + space[idx] + function_code[end:]
        # self.save_function(function_code, count)
        # count += 1
        return function_code
    

    def get_template(self):
        return self._code
    

    def batch_sample(self, batch_size):
        probability = self.calculate_probability()
        indices_list = []
        instance_list = []
        while len(indices_list) < batch_size:
            indices = []
            for prob in probability:
                idx = np.random.choice(len(prob), 1, replace=False, p=prob)
                indices.append(int(idx))
            indices = tuple(indices)
            if indices not in self.visited_indices:
                self.visited_indices.add(indices)
                indices_list.append(indices)
                # instance_list.append(self.get_instance(indices))
                instance_list.append(self)
            else:
                print('repeat sample...')
        return indices_list, instance_list
    

    def update_score(self, instance_indices, score_list):
        for indices, score in zip(instance_indices, score_list):
            for space_i, idx in enumerate(indices):
                if score:
                    self.score_list[space_i][idx] = max(self.score_list[space_i][idx], score)

    
    def get_final_code(self):
        top_20_record = [[] for _ in range(len(self.score_record))] # 记录每个tuable的前20评分的选项(score，idx)
        for idx_,item in enumerate(self.score_record):
            com_list = []
            for idx, scores in enumerate(item):
                com_list += scores
            com_list.sort(key=lambda x: x[0])
            top_20_record[idx_] = com_list[:len(com_list)//5]
        result=[]# 记录评分前20的所有参数组合
        for i in range(len(top_20_record[0])):
            comb = []
            for item in top_20_record:
                comb.append(item[i][1])
            result.append(comb)
        return result

    # def save_function(self, code: str, count: int):
    #     file_name = f"generated_function_{count}.py"
    #     file_path = os.path.join(self._store_folder_name, file_name)
    #     with open(file_path, "w", encoding="UTF-8") as f:
    #         f.write(code)

    # def gen_function(self):
    #     self.gen_function_code()

    # def run(self):
        # os.makedirs(self._store_folder_name, exist_ok=True)
        # with open(self._sample_name, "r", encoding="UTF-8") as file:
        #     code = file.read()
        # self._code = code
        # self.gen_function()
