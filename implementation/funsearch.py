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

"""A single-threaded implementation of the FunSearch pipeline."""
from __future__ import annotations

# from collections.abc import Sequence

# RZ: there are multiple errors in the original code
# we should use typing.xxx rather than collections.abc.xxx
from typing import Any, Tuple, Sequence

from implementation import code_manipulation
from implementation import evaluator
from implementation import programs_database
from implementation import sampler
from implementation import profile
from implementation import sample_iterator


def _extract_function_names(specification: str) -> Tuple[str, str]:
    """Returns the name of the function to evolve and of the function to run.

    RZ: The so-called specification refers to the boilerplate code template for a task.
    The template MUST have two important functions decorated with '@funsearch.run', '@funsearch.evolve' respectively.
    The function labeled with '@funsearch.run' is going to evaluate the generated code (like fitness evaluation).
    The function labeled with '@funsearch.evolve' is the function to be searched (like 'greedy' in cap-set).
    This function (_extract_function_names) makes sure that these decorators appears in the specification.
    """
    run_functions = list(code_manipulation.yield_decorated(specification, 'funsearch', 'run'))
    if len(run_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@funsearch.run`.')
    evolve_functions = list(code_manipulation.yield_decorated(specification, 'funsearch', 'evolve'))
    if len(evolve_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@funsearch.evolve`.')
    return evolve_functions[0], run_functions[0]


def main(
        specification: str,
        inputs: Sequence[Any],
        sandbox_class,
        llm_class,
        max_sample_nums: int | None,
        **kwargs
):
    """Launches a FunSearch experiment.
    RZ:
    Args:
        specification: the boilerplate code for the problem.
        inputs       : the data instances for the problem (see 'bin_packing_utils.py').
        max_sample_nums: the maximum samples nums from LLM. 'None' refers to no stop.
    """
    function_to_evolve, function_to_run = _extract_function_names(specification)
    template = code_manipulation.text_to_program(specification)
    database = programs_database.ProgramsDatabase(template, function_to_evolve)

    # get log_dir and create profiler
    log_dir = kwargs.get('log_dir', None)
    if log_dir is None:
        profiler = None
    else:
        profiler = profile.Profiler(log_dir)

    evaluator_ins = evaluator.Evaluator(
        database,
        template,
        function_to_evolve,
        function_to_run,
        inputs,
        timeout_seconds=30,
        sandbox_class=sandbox_class,
    )

    # We send the initial implementation to be analysed by one of the evaluators.
    initial = template.get_function(function_to_evolve).body
    (sample_template, evaluate_time, score_list, decisions_list), _ = evaluator_ins.analyse(sample_iterator.SampleIterator(initial), [[]])
    profiler.register_function_list(sample_template, None, evaluate_time, score_list, decisions_list)
    new_function, _ = evaluator._sample_to_program(initial, template, function_to_evolve)
    database.register_program(new_function, max(score_list), model=None, parent_score=None)

    # Set global max sample nums.
    sampler_ins = sampler.Sampler(database, template, function_to_evolve, evaluator_ins, max_sample_nums=max_sample_nums, llm_class=llm_class)

    # This loop can be executed in parallel on remote sampler machines. As each
    # sampler enters an infinite loop, without parallelization only the first
    # sampler will do any work.
    sampler_ins.sample(profiler=profiler)
