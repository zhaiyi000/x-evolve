import itertools
import os
import re


class Gen_Func:
    def __init__(
        self, sample_name: str, store_folder_name: str, regular: str, split: str
    ):
        self._code = f""
        self._sample_name = sample_name
        self._store_folder_name = store_folder_name
        self._regular = regular
        self._split = split

    def tunable_find(self):
        pattern_in = self._regular
        matches_in = re.findall(pattern_in, self._code)
        tunable = []
        for item in matches_in:
            tun = tuple(item.split(self._split))
            tunable.append(tun)
        return tunable

    def gen_function_code(self, tunable: itertools.product):
        pattern = self._regular
        matches = list(re.finditer(pattern, self._code))
        all_comb = itertools.product(*tunable)
        count = 0
        for item in all_comb:
            function_code = self._code
            for i in range(len(matches)):
                if re.search(pattern, function_code):
                    match = re.search(pattern, function_code)
                    start, end = match.span()
                    function_code = (
                        function_code[:start] + f"{item[i]}" + function_code[end:]
                    )
            self.save_function(function_code, count)
            count += 1

    def save_function(self, code: str, count: int):
        file_name = f"generated_function_{count}.py"
        file_path = os.path.join(self._store_folder_name, file_name)
        with open(file_path, "w", encoding="UTF-8") as f:
            f.write(code)

    def gen_function(self):
        self.gen_function_code(self.tunable_find())

    def run(self):
        os.makedirs(self._store_folder_name, exist_ok=True)
        with open(self._sample_name, "r", encoding="UTF-8") as file:
            code = file.read()
        self._code = code
        self.gen_function()
