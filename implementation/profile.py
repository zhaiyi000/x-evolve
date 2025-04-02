# implemented by RZ
# profile the experiment using tensorboard

from __future__ import annotations

import os.path
from typing import List, Dict
import logging
import json
from implementation import code_manipulation
from torch.utils.tensorboard import SummaryWriter
import glob


class Profiler:
    def __init__(
            self,
            log_dir: str | None = None,
            pkl_dir: str | None = None,
            max_log_nums: int | None = None,
    ):
        """
        Args:
            log_dir     : folder path for tensorboard log files.
            pkl_dir     : save the results to a pkl file.
            max_log_nums: stop logging if exceeding max_log_nums.
        """
        logging.getLogger().setLevel(logging.INFO)
        self._log_dir = log_dir
        self._json_dir = os.path.join(log_dir, 'samples')
        os.makedirs(self._json_dir, exist_ok=True)
        # self._pkl_dir = pkl_dir
        self._max_log_nums = max_log_nums
        self._cur_best_program_sample_order = None
        self._cur_best_program_score = -99999999
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0
        self._all_sampled_functions: Dict[int, code_manipulation.Function] = {}

        if log_dir:
            self._writer = SummaryWriter(log_dir=log_dir)

        self._each_sample_best_program_score = []
        self._each_sample_evaluate_success_program_num = []
        self._each_sample_evaluate_failed_program_num = []
        self._each_sample_tot_sample_time = []
        self._each_sample_tot_evaluate_time = []
        self._log_file = os.path.join(log_dir, 'profile.log')

        sample_files = glob.glob(os.path.join(self._json_dir, '*.json'))
        print(f'find {len(sample_files)} sample files')
        self._num_samples = len(sample_files)

    def _write_tensorboard(self):
        if not self._log_dir:
            return

        self._writer.add_scalar(
            'Best Score of Function',
            self._cur_best_program_score,
            global_step=self._num_samples
        )
        self._writer.add_scalars(
            'Legal/Illegal Function',
            {
                'legal function num': self._evaluate_success_program_num,
                'illegal function num': self._evaluate_failed_program_num
            },
            global_step=self._num_samples
        )
        self._writer.add_scalars(
            'Total Sample/Evaluate Time',
            {'sample time': self._tot_sample_time, 'evaluate time': self._tot_evaluate_time},
            global_step=self._num_samples
        )

    def _write_json(self, programs: code_manipulation.Function):
        function_str = str(programs)
        score = programs.score
        content = {
            'sample_order': self._num_samples,
            'function': function_str,
            'score': score,
            'decisions': programs.decisions
        }
        path = os.path.join(self._json_dir, f'samples_{self._num_samples}.json')
        with open(path, 'w') as json_file:
            json.dump(content, json_file)

    def register_function(self, programs: code_manipulation.Function):
        if self._max_log_nums is not None and self._num_samples >= self._max_log_nums:
            return

        self._num_samples += 1
        self._all_sampled_functions[self._num_samples] = programs
        self._record_and_verbose(programs)
        self._write_tensorboard()
        self._write_json(programs)

    def _record_and_verbose(self, programs):
        function_str = str(programs).strip('\n')
        sample_time = programs.sample_time
        evaluate_time = programs.evaluate_time
        score = programs.score
        # log attributes of the programs
        # with open(self._log_file, 'a') as f:
        #     f.write(f'================= Evaluated Programs =================\n')
        #     f.write(f'{function_str}\n')
        #     f.write(f'------------------------------------------------------\n')
        #     f.write(f'Score        : {str(score)}\n')
        #     f.write(f'Sample time  : {str(sample_time)}\n')
        #     f.write(f'Evaluate time: {str(evaluate_time)}\n')
        #     f.write(f'Sample orders: {str(self._num_samples)}\n')
        #     f.write(f'Decisions: {str(programs.decisions)}\n')
        #     f.write(f'======================================================\n\n\n')

        # update best programs
        if programs.score is not None and score > self._cur_best_program_score:
            self._cur_best_program_score = score
            self._cur_best_program_sample_order = self._num_samples

        # update statistics about programs
        if score:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1

        if sample_time:
            self._tot_sample_time += sample_time
        if evaluate_time:
            self._tot_evaluate_time += evaluate_time

    
    def register_function_list(self, sample_template, sample_time, evaluate_time, score_list, decisions_list):
        for score, decisions in zip(score_list, decisions_list):
            sample_template.score = score
            sample_template.sample_time = sample_time
            sample_template.evaluate_time = evaluate_time
            sample_template.decisions = decisions
            self.register_function(sample_template)
