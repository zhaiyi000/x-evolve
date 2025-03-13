# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for sampling new programs."""
from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Collection, Sequence, Type
import numpy as np
import time

from implementation import evaluator
from implementation import programs_database
from implementation import sample_iterator
from implementation import code_manipulation
import copy
import threading
import queue
from implementation import sample_llm_api


class LLM(ABC):
    """Language model that predicts continuation of provided source code.

    RZ: The sampled function code must be trimmed! Especially using instruct-based LLM.
    -For example, the sampled function code (with description) is:
    ------------------------------------------------------------------------------------------------------------------
    Here is the function.
    def priority_v2(..., ...) -> Any:
        a = np.array([1, 2, 3])
        if len(a) > 2:
            return a / a.sum()
        else:
            return a / a.mean()
    This function is going to ..., and returns ...[Descriptions by LLM]
    ------------------------------------------------------------------------------------------------------------------
    -The descriptions above the function's signature, and the function's signature must be removed.
    -The above code must be trimmed as follows:
    ------------------------------------------------------------------------------------------------------------------
        a = np.array([1, 2, 3])
            if len(a) > 2:
                return a / a.sum()
            else:
                return a / a.mean()
        Here is the function. This function is going to ..., and returns ...[Descriptions by LLM]
    ------------------------------------------------------------------------------------------------------------------
    Please note that the indent must be preserved. And the additional descriptions can also be preserved,
    which will be trimmed by Evaluator.
    """

    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        """Returns a predicted continuation of `prompt`."""
        raise NotImplementedError('Must provide a language model.')

    @abstractmethod
    def draw_samples(self, llm_ins, prompt: str) -> Collection[str]:
        """Returns multiple predicted continuations of `prompt`."""
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]


class Sampler:
    """Node that samples program continuations and sends them for analysis.
    """
    _global_samples_nums: int = 1  # RZ: this variable records the global sample nums
    _global_spaces_nums: int = 1

    def __init__(
            self,
            database: programs_database.ProgramsDatabase,
            template: code_manipulation.Program,
            function_to_evolve: str,
            evaluator: evaluator.Evaluator,
            samples_per_prompt: int,
            max_sample_nums: int | None = None,
            llm_class: Type[LLM] = LLM
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluator = evaluator
        self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._mux_sem = threading.Semaphore(1)
        self._queue = queue.Queue()
        self._llm_cnt = 10


    def launch_llm(self, thread_i, llm):
        try:
            while True:
                print('current thread_i', thread_i)
                with self._mux_sem:
                    # stop the search process if hit global max sample nums
                    if self._max_sample_nums and self.__class__._global_spaces_nums >= self._max_sample_nums:
                        self._queue.put('end')
                        break
                    # try:
                    prompt, parent_score = self._database.get_prompt()
                    llm_ins = llm.calculate_probability()
                reset_time = time.time()
                samples = self._llm.draw_samples(llm_ins, prompt.code)
                sample_time = (time.time() - reset_time) / self._samples_per_prompt
                self._queue.put((samples, sample_time, llm_ins, parent_score))
                time.sleep(0.1)
        except Exception as err:
            print('current thread_i', thread_i)
            print('errrrrrr', 'launch_llm', err)

    def update_database(self, samples, sample_time, llm_ins, parent_score, llm, kwargs):
        # samples_new = []
        for sample in samples:
            print('-------------------')
            print(sample)
            print(f'call llm times: {self._get_global_spaces_nums()}')
            print('-------------------\n\n')
            self._global_spaces_nums_plus_one()
            tune_sampler = sample_iterator.SampleIterator(code=sample)
            batch_size = 64
            MIN_SCORE = -1e10
            max_score = MIN_SCORE
            while True:
                indices = tune_sampler.batch_sample(batch_size=batch_size)

                num_list = []
                for _ in indices:
                    self._global_sample_nums_plus_one()  # RZ: add _global_sample_nums
                    cur_global_sample_nums = self._get_global_sample_nums()
                    num_list.append(cur_global_sample_nums)
                
                score_list = self._evaluator.analyse(
                    tune_sampler,
                    indices,
                    # prompt.version_generated,
                    **kwargs,
                    global_sample_nums_list=num_list,
                    sample_time=sample_time
                )
                # score_list = [max_score, *[x for x in score_list if x]]
                max_score = max([max_score, *[x for x in score_list if x]])
                
                if tune_sampler.update_score(indices, score_list) is False:
                    print('sampler suggest should end sample, break')
                    break

            if max_score != MIN_SCORE:
                function_code = tune_sampler.get_final_code()
                new_function = evaluator._sample_to_program(
                    function_code, self._template, self._function_to_evolve)

                self._database.register_program(
                    new_function,
                    max_score,
                )
            llm.call_llm(llm_ins, parent_score, max_score)


    def sample(self, **kwargs):
        """Continuously gets prompts, samples programs, sends them for analysis.
        """
        llm = sample_llm_api.EvaluateLLM()
        for thread_i in range(self._llm_cnt):
            launch_thread = threading.Thread(target=self.launch_llm, args=(thread_i, llm), daemon=True)
            launch_thread.start()

        while True:
            llm_return_obj = self._queue.get()
            if llm_return_obj == 'end':
                break
            with self._mux_sem:
                samples, sample_time, llm_ins, parent_score = llm_return_obj
                self.update_database(samples, sample_time, llm_ins, parent_score, llm, kwargs)
                

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1


    def _get_global_spaces_nums(self) -> int:
        return self.__class__._global_spaces_nums

    def set_global_spaces_nums(self, num):
        self.__class__._global_spaces_nums = num

    def _global_spaces_nums_plus_one(self):
        self.__class__._global_spaces_nums += 1
