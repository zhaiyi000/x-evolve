import itertools
import os
import re


FILE_DEBUG_MODE = False


class SampleIterator:
    def __init__(
        self, code: str
    ):
        self._code = code
        # self._sample_name = sample_name
        # self._store_folder_name = store_folder_name
        self._regular = "tunable\(\[(.*?)\]\)"
        self._split = ','

    # def tunable_find(self):
    #     matches_in = re.findall(self._regular, self._code)
    #     tunable = []
    #     for item in matches_in:
    #         tun = tuple(item.split(self._split))
    #         tunable.append(tun)
    #     return tunable

    def iterate_sample(self):
        matches = list(re.finditer(self._regular, self._code))
        matches_update = False
        tunable = []
        for match in reversed(matches):
            options = match.group(1).split(self._split)
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
        all_comb = itertools.product(*tunable)
        # count = 0
        instances = []
        for comb in all_comb:
            function_code = self._code
            for match, item in zip(reversed(matches), reversed(comb)):
                start, end = match.span()
                function_code = function_code[:start] + item + function_code[end:]
            # self.save_function(function_code, count)
            # count += 1
            instances.append(function_code)
            # if 'tunable(' in function_code:
            #     raise Exception('tuneable in function_code')
        return instances

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
