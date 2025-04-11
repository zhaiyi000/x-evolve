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

"""A programs database that implements the evolutionary algorithm."""
from __future__ import annotations

import profile
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import time
from typing import Any, Tuple, Mapping

import natsort.natsort
import numpy as np
import scipy

from implementation import code_manipulation
from implementation import evaluate_function
import heapq, queue, math
from config import log_dir
import os
import json
import natsort, glob
from config import sample_llm_api_min_score

# RZ: I change the original code "tuple[float, ...]" to "Tuple[float, ...]"
Signature = Tuple[float, ...]

# RZ: the code is also incorrect
# We should use typing.Mapping rather than abc.Mapping
ScoresPerTest = Mapping[Any, float]


# def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
#     """Returns the tempered softmax of 1D finite `logits`."""
#     if not np.all(np.isfinite(logits)):
#         non_finites = set(logits[~np.isfinite(logits)])
#         raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
#     if not np.issubdtype(logits.dtype, np.floating):
#         logits = np.array(logits, dtype=np.float32)

#     result = scipy.special.softmax(logits / temperature, axis=-1)
#     # Ensure that probabilities sum to 1 to prevent error in `np.random.choice`.
#     index = np.argmax(result)
#     result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1:])
#     return result


def reduce_score(scores_per_test: ScoresPerTest) -> float:
    """Reduces per-test scores into a single score.
    """
    # TODO RZ: change the code to average the score of each test.
    # return scores_per_test[list(scores_per_test.keys())[-1]]
    test_scores = [scores_per_test[k] for k in scores_per_test.keys()]
    return sum(test_scores) / len(test_scores)


# def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
#     """Represents test scores as a canonical signature."""
#     return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))


@dataclasses.dataclass(frozen=True)
class Prompt:
    """A prompt produced by the ProgramsDatabase, to be sent to Samplers.

    Attributes:
      code: The prompt, ending with the header of the function to be completed.
      version_generated: The function to be completed is `_v{version_generated}`.
      island_id: Identifier of the island that produced the implementations
         included in the prompt. Used to direct the newly generated implementation
         into the same island.
    """
    code: str
    version_generated: int

@dataclasses.dataclass
class Node:
    visit_count: int
    score: float
    # score_sum: float
    # score_update: float
    # node_id: int
    program: code_manipulation.Function
    model: str
    parent_score: list[float]

    # @property
    # def score_avg(self):
    #     return self.score_sum / self.visit_count


class ProgramsDatabase:
    """A collection of programs, organized as islands."""

    def __init__(
            self,
            template: code_manipulation.Program,
            function_to_evolve: str,
    ) -> None:
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve
        self._functions_per_prompt: int = 2
        self._nodes: list[Node] = []
        self._best_score: float = -float('inf')

        node_dir = os.path.join(log_dir, 'node')
        node_files = glob.glob(os.path.join(node_dir, '*.json'))
        node_files = natsort.natsorted(node_files)
        for file in node_files:
            with open(file, 'r') as f:
                data = json.load(f)
            program = code_manipulation.Function(**data['program'])
            if data['score'] > sample_llm_api_min_score:
                node = Node(visit_count=data['visit_count'], score=data['score'], program=program, model=data['model'], parent_score=data['parent_score'])
                self._nodes.append(node)
            else:
                print(f'find less equal min score {data["score"]}')
        self.save_idx = 0
        if len(node_files) > 0:
            self.save_idx = int(os.path.splitext(os.path.basename(node_files[-1]))[0]) + 1
        print(f'find {len(node_files)} node files')


    def get_prompt(self) -> Prompt:
        nodes = self._nodes
        score_list = [node.score for node in nodes]
        visit_list = [node.visit_count for node in nodes]
        length_list = [len(str(node.program)) for node in nodes]
        
        functions_per_prompt = min(len(self._nodes), self._functions_per_prompt)
        best_indices = evaluate_function.calculate_score(score_list=score_list, size=functions_per_prompt, replace=True)
        best_nodes = [nodes[i] for i in best_indices]
        best_nodes = list(best_nodes)

        best_nodes.sort(key=lambda x: x.score)

        sorted_implementations = []
        parent_score = []
        for node in best_nodes:
            node.visit_count += 1
            sorted_implementations.append(node.program)
            parent_score.append(node.score)

        version_generated = len(sorted_implementations) + 1
        code = self._generate_prompt(sorted_implementations)
        return Prompt(code, version_generated), parent_score

    def _generate_prompt(
            self,
            implementations: Sequence[code_manipulation.Function]) -> str:
        """Creates a prompt containing a sequence of function `implementations`."""
        implementations = copy.deepcopy(implementations)  # We will mutate these.

        # Format the names and docstrings of functions to be included in the prompt.
        versioned_functions: list[code_manipulation.Function] = []
        for i, implementation in enumerate(implementations):
            new_function_name = f'{self._function_to_evolve}_v{i}'
            implementation.name = new_function_name
            # Update the docstring for all subsequent functions after `_v0`.
            if i >= 1:
                implementation.docstring = (
                    f'Improved version of `{self._function_to_evolve}_v{i - 1}`.')
            # If the function is recursive, replace calls to itself with its new name.
            implementation = code_manipulation.rename_function_calls(
                str(implementation), self._function_to_evolve, new_function_name)
            versioned_functions.append(
                code_manipulation.text_to_function(implementation))

        # Create the header of the function to be generated by the LLM.
        next_version = len(implementations)
        new_function_name = f'{self._function_to_evolve}_v{next_version}'
        header = dataclasses.replace(
            implementations[-1],
            name=new_function_name,
            body='',
            docstring=('Improved version of '
                       f'`{self._function_to_evolve}_v{next_version - 1}`.'),
        )
        versioned_functions.append(header)

        # Replace functions in the template with the list constructed here.
        prompt = dataclasses.replace(self._template, functions=versioned_functions)
        return str(prompt)


    def register_program(
            self,
            program: code_manipulation.Function,
            scores_per_test: ScoresPerTest,
            model,
            parent_score
            # **kwargs  # RZ: add this for profiling
    ) -> None:
        """Registers `program` in the specified island."""
        if isinstance(scores_per_test, (np.float64, float, int)):
            score = scores_per_test
        elif isinstance(scores_per_test, ScoresPerTest):
            raise Exception('todo')
            score = reduce_score(scores_per_test)
        else:
            raise Exception('unkonw data type')

        node = Node(visit_count=0, score=score, program=program, model=model, parent_score=parent_score)
        self._nodes.append(node)

        node_dir = os.path.join(log_dir, 'node')
        os.makedirs(node_dir, exist_ok=True)
        node_file = os.path.join(node_dir, f'{self.save_idx}.json')
        self.save_idx += 1
        with open(node_file, 'w') as f:
            json.dump(dataclasses.asdict(node), f)

        if score > self._best_score:
            self._best_score = score
            print('Best score increased to %s' % score)
