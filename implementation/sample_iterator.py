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

    max_score_global = MIN_SCORE

    def __init__(
        self, code: str
    ):
        self._code = code
        # self._final_code = None
        # self._sample_name = sample_name
        # self._store_folder_name = store_folder_name
        self._regular = "tunable\(\[(.*?)\]\)"
        self._split = ','
        self._temperature = 1

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
        self.visited = {}
        self.tunable = tunable
        self.matches = matches
        self.best_score = MIN_SCORE
        self.no_update_cnt = 0
        self.space_size = np.prod([len(space) for space in tunable])


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
    

    # def get_template(self):
    #     return self._code
    

    def batch_sample(self, batch_size):
        probability = self.calculate_probability()
        indices_list = []
        instance_list = []
        try_cnt = 0
        while len(indices_list) < batch_size and try_cnt < batch_size:
            indices = []
            for prob in probability:
                idx = np.random.choice(len(prob), 1, replace=False, p=prob)
                indices.append(int(idx))
            indices = tuple(indices)
            if indices not in self.visited:
                self.visited[indices] = MIN_SCORE
                indices_list.append(indices)
                # instance_list.append(self.get_instance(indices))
                instance_list.append(self)
                try_cnt = 0
            else:
                try_cnt += 1
                print('.', end='')
        print()
        return indices_list, instance_list
    

    def update_score(self, instance_indices, score_list):
        best_score = MIN_SCORE
        for indices, score in zip(instance_indices, score_list):
            if score:
                best_score = max(best_score, score)
                assert indices in self.visited
                self.visited[indices] = score
                for space_i, idx in enumerate(indices):
                    self.score_list[space_i][idx] = max(self.score_list[space_i][idx], score)
        if best_score > self.best_score:
            self.__class__.max_score_global = max(self.__class__.max_score_global, best_score)
            self.best_score = best_score
            self.no_update_cnt = 0
        else:
            self.no_update_cnt += 1

        print(f'this best socre: {best_score}; best score: {self.best_score}; global score: {self.__class__.max_score_global}; space size: {self.space_size}; measure cnt: {len(self.visited)}')
        factor = 4 if self.best_score == self.__class__.max_score_global else 1
        if self.space_size == len(self.visited) or self.no_update_cnt == 3 * 1:
            return False
        else:
            return True

    
    def get_final_code(self):
        top_cnt = 1
        reocrds = list(self.visited.items())
        np.random.shuffle(reocrds)
        reocrds.sort(key=lambda x: x[1], reverse=True)
        reocrds = reocrds[:top_cnt]
        indices_list = [x[0] for x in reocrds]
        function_code = self._code
        space_i = len(self.tunable) - 1
        for match, space in zip(reversed(self.matches), reversed(self.tunable)):
            idx_set = set()
            for indices in indices_list:
                idx_set.add(indices[space_i])
            start, end = match.span()
            if len(idx_set) == 0:
                raise Exception('len idx set equal 0')
            elif len(idx_set) == 1:
                replace_str = space[list(idx_set)[0]]
            else:
                idx_list = list(idx_set)
                idx_list.sort()
                replace_str = 'tunable(['
                for idx_i, idx in enumerate(idx_list):
                    if idx_i != 0:
                        replace_str += ', '
                    replace_str += space[idx]
                replace_str += '])'
            function_code = function_code[:start] + replace_str + function_code[end:]
            space_i -= 1
        return function_code

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
