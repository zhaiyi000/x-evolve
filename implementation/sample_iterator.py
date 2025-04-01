import itertools
import os
import re
# import random
import numpy as np
import libcst as cst
from libcst.metadata import PositionProvider
from redbaron import *
from config import *


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

TUNABLE = 'tunable'
SAMPLE_REGULAR = f"{TUNABLE}\(\[(.*?)\]\)"
SPLIT_CHAR = ','

def parse_tunables_with_comments(source_code):
    """解析源代码并保留注释/格式，找出tunable参数"""
    # 大模型返回的程序源代码
    with open('orginal_code.log','a+') as file:
        file.write(source_code)
        file.write('\n')
    module = cst.parse_module(source_code)
    tunables = []
    tunables_all = []
    _module = cst.Module(body=[])
    class TunableCollector(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (PositionProvider,)
        
        def visit_Call(self, node):
            if isinstance(node.func, cst.Name) and node.func.value == "tunable":
                if node.args and isinstance(node.args[0].value, cst.List):
                    list_elements = node.args[0].value.elements
                    pos = self.get_metadata(PositionProvider, node).start
                    tunables_all.append(_module.code_for_node(node.args[0]))
                    tunables.append({
                        "lineno": pos.line,
                        "column": pos.column,
                        "length": len(list_elements),
                        "node": node
                    })
    
    wrapper = cst.MetadataWrapper(module)
    collector = TunableCollector()
    wrapper.visit(collector)
    return tunables, module ,tunables_all

# 比较标志
def is_in(str_1:str,str_2:str):
    if str_1 in str_2:
        _str_2 = str_2.split('=')
        for item in _str_2:
            clear_item = item.strip()
            if str_1 == clear_item:
                return True
    if 'tunable' in str_1:
        clear_str_1 = re.sub(SAMPLE_REGULAR,'tunable',str_1)
        _str_1 = clear_str_1.split('tunable')
        for item in _str_1:
            if item not in str_2:
                return False
        return True
    return False

class SampleIterator:

    max_score_global = MIN_SCORE

    def __init__(
        self, code: str
    ):
        self._initial_code = code
        if 'def priority' in code:
            self._code = code
        else:
            self._code = 'def priority(item: float, bins: np.ndarray) -> np.ndarray:' + '\n' + code
        # self._regular = SAMPLE_REGULAR
        # self._split = SPLIT_CHAR
        self._temperature = 1
        self._tunables, self._module, self._tunables_all= parse_tunables_with_comments(self._code)
        # matches = list(re.finditer(self._regular, self._code))
        # matches_update = False
        # tunable = []
        # for match in reversed(matches):
        #     options = match.group(1).split(self._split)
        #     options = [x.strip() for x in options]
        #     if len(options) == 0:
        #         raise Exception('options\'s len is zero')
        #     elif len(options) == 1:
        #         start, end = match.span()
        #         self._code = self._code[:start] + options[0] + self._code[end:]
        #         matches_update = True
        #     else:
        #         tunable.append(options)
        # if matches_update:
        #     matches = list(re.finditer(self._regular, self._code))
        # tunable.reverse()

        self.score_list = [[MIN_SCORE] * space['length'] for space in self._tunables]
        self.visited = {}
        # self.tunable = tunable
        # self.matches = matches
        self.best_score = MIN_SCORE
        self.no_update_cnt = 0
        self.space_size = np.prod([space['length'] for space in self._tunables])


    def calculate_probability(self):
        probability = []
        for scores in self.score_list:
            max_score = max(scores)
            scores = [x if x != MIN_SCORE else max_score for x in scores]
            prob = softmax(scores, self._temperature)
            probability.append(prob)
        return probability
    

    # def get_instance(self, indices):
    #     function_code = self._code
    #     decisions = []
    #     for match, space, idx in zip(reversed(self.matches), reversed(self.tunable), reversed(indices)):
    #         start, end = match.span()
    #         function_code = function_code[:start] + space[idx] + function_code[end:]
    #         decisions.append(space[idx])
    #     decisions.reverse()
    #     return function_code, decisions
    
    def replace_tunables_with_comments(self, module, tunables_info, replace_indices):
        """替换tunable参数并保持原始格式"""
        decisions = []
        class TunableReplacer(cst.CSTTransformer):
            def __init__(self):
                self.current_index = 0
                self._module = cst.Module(body=[])
                
            def leave_Call(self, original_node, updated_node):
                nonlocal replace_indices, decisions
                if isinstance(updated_node.func, cst.Name) and updated_node.func.value == "tunable":
                    if self.current_index < len(replace_indices):
                        replace_idx = replace_indices[self.current_index]
                        self.current_index += 1
                        if updated_node.args and isinstance(updated_node.args[0].value, cst.List):
                            list_elements = updated_node.args[0].value.elements
                            # print(list_elements)
                            if 0 <= replace_idx < len(list_elements):
                                decisions.append(self._module.code_for_node(list_elements[replace_idx].value))
                                return  list_elements[replace_idx].value
                return updated_node
        return module.visit(TunableReplacer()), decisions
    
    def get_instance(self, indices):
        modified_module,decisions= self.replace_tunables_with_comments(
            module = self._module,
            tunables_info = self._tunables,
            replace_indices = indices
        )
        # RedBaron 找到所有形如'# 注释'的注释
        red = RedBaron(self._code)
        comments=[]
        for comment in red.find_all('comment'):
            comments.append({
                'text':comment.value.strip(),
            })
        # 找到注释所对应的行及行内标志信息
        lines = self._code.splitlines()
        for index,line in enumerate(lines):
            for item in comments:
                if item['text'] in line:
                    item['sign'] = line.replace(item['text'],'').strip().strip(',')
                    item['line'] = index
        # 仅保留所需注释
        comments_clear = []
        for item in comments:
            if len(item['sign']) != 0:
                comments_clear.append(item)
                # print(item)
        # 注释回填
        mm_lines = modified_module.code.splitlines()
        final_code = []
        for index,line in enumerate(mm_lines):
            for item in comments_clear:
                if is_in(item['sign'],line) and item['line'] >= index:
                    line = line + ' ' + item['text']
                    item['line'] = -1
            final_code.append(line)
        # print(comments_clear)
                    
        f_code = ''
        for item in final_code[1:]:
            f_code = f_code + item + '\n'
        
        return f_code, decisions
        

    # def get_instance_by_decisions(self, decisions):
    #     if len(self.matches) != len(decisions):
    #         print('❗️', end='')
    #     function_code = self._code
    #     for match, space, decision in zip(reversed(self.matches), reversed(self.tunable), reversed(decisions)):
    #         start, end = match.span()
    #         if decision not in space:
    #             print('x', end='')
    #         function_code = function_code[:start] + decision + function_code[end:]
    #     return function_code
    def decisions_replace(self, module, tunables_info, decisions):
        class TunableReplacer(cst.CSTTransformer):
            def __init__(self):
                self.current_index = 0
                
            def leave_Call(self, original_node, updated_node):
                nonlocal decisions
                if isinstance(updated_node.func, cst.Name) and updated_node.func.value == "tunable":
                    if self.current_index < len(decisions):
                        self.current_index += 1
                        if updated_node.args and isinstance(updated_node.args[0].value, cst.List):
                            return  decisions[self.current_index-1]
                return updated_node
        return module.visit(TunableReplacer())
    
    def decisions_code(self, decisions):
        modified_module= self.decisions_replace(
            module = self._module,
            tunables_info = self._tunables,
            replace_indices = decisions
        )
        # RedBaron 找到所有形如'# 注释'的注释
        red = RedBaron(self._code)
        comments=[]
        for comment in red.find_all('comment'):
            comments.append({
                'text':comment.value.strip(),
            })
        # 找到注释所对应的行及行内标志信息
        lines = self._code.splitlines()
        for index,line in enumerate(lines):
            for item in comments:
                if item['text'] in line:
                    item['sign'] = line.replace(item['text'],'').strip().strip(',')
                    item['line'] = index
        # 仅保留所需注释
        comments_clear = []
        for item in comments:
            if len(item['sign']) != 0:
                comments_clear.append(item)
                # print(item)
        # 注释回填
        mm_lines = modified_module.code.splitlines()
        final_code = []
        for index,line in enumerate(mm_lines):
            for item in comments_clear:
                if is_in(item['sign'],line) and item['line'] >= index:
                    line = line + ' ' + item['text']
                    item['line'] = -1
            final_code.append(line)
        # print(comments_clear)
                    
        f_code = ''
        for item  in final_code[1:]:
            f_code = f_code + item + '\n'
        
        return f_code
    
    def get_instance_by_decisions(self, decisions):
        if len(self._tunables) != len(decisions):
            print('❗️', end='')
        for i in decisions:
            cnt = 0
            for item in self._tunables_all:
                if i in item:
                    cnt += 1
            if cnt == 0:
                print('x', end='')
        _node = []
        for item in decisions:
            _node.append(cst.parse_expression(item))
        function_code = self.decisions_code(_node)
        return function_code
    

    def get_template(self):
        return self._initial_code
    

    def batch_sample(self, batch_size):
        probability = self.calculate_probability()
        indices_list = []
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
                try_cnt = 0
            else:
                try_cnt += 1
                print('.', end='')
        print()
        return indices_list
    

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
        if self.space_size == len(self.visited) or self.no_update_cnt == sample_iterator_no_update_cnt * 1:
            return False
        else:
            return True

    
    # def get_final_code(self):
    #     top_cnt = 1
    #     reocrds = list(self.visited.items())
    #     np.random.shuffle(reocrds)
    #     reocrds.sort(key=lambda x: x[1], reverse=True)
    #     reocrds = reocrds[:top_cnt]
    #     indices_list = [x[0] for x in reocrds]
    #     function_code = self._code
    #     space_i = len(self.tunable) - 1
    #     for match, space in zip(reversed(self.matches), reversed(self.tunable)):
    #         idx_set = set()
    #         for indices in indices_list:
    #             idx_set.add(indices[space_i])
    #         start, end = match.span()
    #         if len(idx_set) == 0:
    #             raise Exception('len idx set equal 0')
    #         elif len(idx_set) == 1:
    #             replace_str = space[list(idx_set)[0]]
    #         else:
    #             idx_list = list(idx_set)
    #             idx_list.sort()
    #             replace_str = 'tunable(['
    #             for idx_i, idx in enumerate(idx_list):
    #                 if idx_i != 0:
    #                     replace_str += ', '
    #                 replace_str += space[idx]
    #             replace_str += '])'
    #         function_code = function_code[:start] + replace_str + function_code[end:]
    #         space_i -= 1
    #     return function_code
    
    def get_final_code(self):
        top_cnt = 1
        records = list(self.visited.items())
        np.random.shuffle(records)
        records.sort(key=lambda x: x[1], reverse=True)
        records = records[:top_cnt]
        indices = records[top_cnt-1][0]
        f_code = self.get_instance(indices=indices)[0]
        return f_code