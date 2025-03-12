from dataclasses import dataclass, field
from transformers import HfArgumentParser
from tokenizer import Tokenizer
from datasets import load_dataset
import json
import numpy as np


@dataclass
class ScriptArguments:
    trainer_class: str = field(metadata={"help": ""})


def make_dataset(file, dataset_path, tokenizer_path, valid_percentage=5):
    data_files = {}
    data_files["train"] = file
    extension = data_files["train"].split(".")[-1]
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        keep_in_memory=True
    )
    if valid_percentage > 0:
        raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{valid_percentage}%]",
            keep_in_memory=True
        )
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{valid_percentage}%:]",
            keep_in_memory=True
        )
    else:
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train",
            keep_in_memory=True
        )

    tokenizer = Tokenizer.from_pretrained(tokenizer_path)

    if script_args.trainer_class == 'trainer':
        pass
    else:
        with open('json/body_visit_weight.json', 'r') as f:
            visit_weight_dic = json.load(f)
        with open('json/body_rewards.json', 'r') as f:
            rewards_dic = json.load(f)

    column_names = list(raw_datasets["train"].features)
    def tokenize_function(examples):
        output = tokenizer(examples["text"], padding='max_length')

        input_ids = output['input_ids']
        attention_mask = output['attention_mask']

        output['input_ids'] = np.array(input_ids, dtype=np.int32)
        output['attention_mask'] = np.array(attention_mask, dtype=np.int8)

        if script_args.trainer_class == 'trainer':
            output["rewards"] = examples['rewards']
        else:
            output["labels"] = output['input_ids'].copy()
            output["labels"][output["labels"]==tokenizer.pad_token_id] = -100

            visit_weight_list = []
            for ids in input_ids:
                visit_weight = []
                ids_tmp = []
                weight_prod = 1
                for i in range(len(ids)-1):
                    if ids[i+1] == tokenizer.pad_token_id:
                        # visit_weight.append(0)
                        break
                    else:
                        ids_tmp.append(ids[i])
                        weight_prod *= visit_weight_dic[str(ids_tmp)]
                        visit_weight.append(weight_prod)
                visit_weight_list.append(visit_weight)

            output["visit_weight"] = visit_weight_list

            rewards_idx_list = []
            rewards_value_list = []
            for ids in input_ids:
                rewards_idx = []
                rewards_value = []
                ids_tmp = []
                for i in range(len(ids)-1):
                    if ids[i+1] == tokenizer.pad_token_id:
                        # rewards_idx.append([])
                        # rewards_value.append([])
                        break
                    else:
                        ids_tmp.append(ids[i])
                        labels = rewards_dic[str(ids_tmp)]
                        reward_index = []
                        reward_value = []
                        for token_id, info in labels.items():
                            reward_index.append(int(token_id))
                            reward_value.append(info['reward'])
                        rewards_idx.append(reward_index)
                        rewards_value.append(reward_value)
                rewards_idx_list.append(rewards_idx)
                rewards_value_list.append(rewards_value)

            output["rewards_idx"] = rewards_idx_list
            output["rewards_value"] = rewards_value_list
        

        return output

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on every text in dataset",
        keep_in_memory=True
    )

    tokenized_datasets.save_to_disk(dataset_path)


if __name__ == '__main__':
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)
    make_dataset('json/body.json', 'dataset_value', 'tokenizer', valid_percentage=20)