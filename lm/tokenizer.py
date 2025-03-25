
import json
import re
import multiprocessing
from datasets import Dataset, DatasetDict
import os
import glob
import natsort
from concurrent.futures import ProcessPoolExecutor
from implementation import sample_iterator
import numpy as np

SPLIT_CHARS = '\{\}()[]\t\n: ,\'".+-=*/~|^?'
PAD_TOKEN = '[PAD]'
ANS_TOKEN = '[ANS]'
SEP_TOKEN = '[SEP]'
MASK_TOKEN = '[MASK]'
SPLIT_CHARS_RE = f'[{re.escape(SPLIT_CHARS)}]'
PAD_CHAR_RE = f'{re.escape(PAD_TOKEN)}'
ANS_CHAR_RE = f'{re.escape(ANS_TOKEN)}'
SEP_CHAR_RE = f'{re.escape(SEP_TOKEN)}'
MASK_CHAR_RE = f'{re.escape(MASK_TOKEN)}'
FINDALL_RE = f'{PAD_CHAR_RE}|{ANS_CHAR_RE}|{SEP_CHAR_RE}|{MASK_CHAR_RE}|{SPLIT_CHARS_RE}|[^{SPLIT_CHARS_RE[1:-1]}]+'
SPECIAL_TOKENS = [PAD_TOKEN, ANS_TOKEN, SEP_TOKEN, MASK_TOKEN]


def tokenizer_encode_inner(vocab, pad_token_id, ans_token_id, sep_token_id, function_list):
    try:
        segent = []
        for function in function_list:
            if len(function) == 1:
                function = function[0]
                decision = None
            elif len(function) == 2:
                function, decision = function
            else:
                raise Exception('wrong function len')
            words = []
            parts = re.split(sample_iterator.SAMPLE_REGULAR, function)
            for part_i, part in enumerate(parts):
                if part_i % 2 == 0:
                    words += re.findall(FINDALL_RE, part)
                else:
                    items = part.split(sample_iterator.SPLIT_CHAR)
                    items = [x.strip() for x in items]
                    words += items

            ids = [vocab[word] for word in words]
            ids.append(ans_token_id)
            labels = None
            if decision:
                labels = [vocab[word] for word in decision]
                # labels.append(sep_token_id)

            segent.append((ids, labels))
        return segent
    except Exception as err:
        raise err


def tokenizer_decode_inner(id_to_token, ids, add_special_tokens, special_tokens):
    try:
        if add_special_tokens:
            return ''.join([id_to_token(x) for x in ids])
        else:
            return ''.join([id_to_token(x) for x in ids if x not in special_tokens])
    except Exception as err:
        raise err


class Tokenizer():
    def __init__(self, dir_path=None):
        if dir_path:
            path = os.path.join(dir_path, 'tokenizer.json')
            with open(path, 'r') as f:
                tokenizer_dic = json.load(f)

            self._vocab = tokenizer_dic['vocab']
            
            id_to_token_dic = list(self._vocab.items())
            id_to_token_dic.sort(key=lambda x: x[1])
            self._id_to_token = [x[0] for x in id_to_token_dic]
            
            self._model_max_length = tokenizer_dic['model_max_length']
            self._pad_token_id = self._vocab[PAD_TOKEN]
            self._ans_token_id = self._vocab[ANS_TOKEN]
            self._sep_token_id = self._vocab[SEP_TOKEN]
            self._mask_token_id = self._vocab[MASK_TOKEN]
            self.special_tokens = [self._pad_token_id, self._ans_token_id, self._sep_token_id]
            self.executor = None


    # def encode(self, sentence, padding=None):
    #     model_max_length = self._model_max_length if padding == 'max_length' else None
    #     ids, mask = tokenizer_encode_inner(self.vocab, self._pad_token_id, self._cls_token_id, self._sep_token_id, model_max_length, sentence, None)
    #     return ids


    def batch_encode(self, function_list, padding):
        if len(function_list) == 0:
            raise Exception('todo')
        else:
            max_workers=min(os.cpu_count(), 64)
            if self.executor is None:
                self.executor = ProcessPoolExecutor(max_workers=max_workers)

            worker_len = (len(function_list) + max_workers - 1) // max_workers
            futures = []
            for start in range(0, len(function_list), worker_len):
                future = self.executor.submit(tokenizer_encode_inner, self._vocab, self._pad_token_id, self._ans_token_id, self._sep_token_id,
                                              function_list[start:start+worker_len])
                futures.append(future)

            result_list = []
            for future in futures:
                result = future.result()
                result_list.append(result)
            return [x for segment in result_list for x in segment]
    

    def decode(self, ids, add_special_tokens=True):
        return tokenizer_decode_inner(self.id_to_token, ids, add_special_tokens, self.special_tokens)
    

    # def batch_decode(self, input_ids, add_special_tokens=True):
    #     if len(input_ids) == 0:
    #         return []
    #     # elif len(input_ids) == 1:
    #     else:
    #         sentence_list = []
    #         for ids in input_ids:
    #             sentence = tokenizer_decode_inner(self.id_to_token, ids, None, add_special_tokens, self.special_tokens)
    #             sentence_list.append(sentence)
    #         return sentence_list
        # else:
        #     err_queue = multiprocessing.Manager().Queue()
        #     pool_results = []
        #     with multiprocessing.Pool(min(len(input_ids), multiprocessing.cpu_count())) as pool:
        #         for ids in input_ids:
        #             result = pool.apply_async(tokenizer_decode_inner, args=(self.id_to_token, ids, err_queue, add_special_tokens, self.special_tokens))
        #             pool_results.append(result)
        #         pool.close()
        #         pool.join()

        #     if err_queue.empty() is False:
        #         err = err_queue.get()
        #         raise err

        #     sentence_list = []
        #     for result in pool_results:
        #         sentence = result.get()
        #         sentence_list.append(sentence)
        #     return sentence_list
    

    def __call__(self, function_list, padding=None):
        return self.batch_encode(function_list, padding=padding)


    @property
    def model_max_length(self):
        return self._model_max_length
    
    @property
    def pad_token_id(self):
        return self._pad_token_id
    
    @property
    def ans_token_id(self):
        return self._ans_token_id
    
    @property
    def sep_token_id(self):
        return self._sep_token_id
    
    @property
    def mask_token_id(self):
        return self._mask_token_id

    # @property
    # def eos_token_id(self):
    #     return self._sep_token_id

    @property
    def vocab_size(self):
        return len(self._vocab)

    @classmethod
    def from_pretrained(cls, path):
        return Tokenizer(path)
    
    def id_to_token(self, id):
        return self._id_to_token[id]
    

    def save_pretrained(self, dir_path, vocab_list=None, model_max_length=None):
        if vocab_list is not None:
            vocab = {}
            for item_i, item in enumerate(vocab_list):
                vocab[item] = item_i
        elif hasattr(self, '_vocab'):
            vocab = self._vocab
        else:
            raise Exception('vocab is invalid')
        
        if model_max_length is None and hasattr(self, '_model_max_length'):
            model_max_length = self._model_max_length

        tokenizer_dic = dict(vocab=vocab, model_max_length=model_max_length)

        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, 'tokenizer.json')
        with open(path, 'w') as f:
            json.dump(tokenizer_dic, f, indent=2)


def test_model_max_length(function_list, tokenizer_path):
    tokenizer = Tokenizer.from_pretrained(tokenizer_path)
    tokens_list = tokenizer(function_list)
    model_max_length = max(len(tokens[0])+len(tokens[1]) for tokens in tokens_list)
    print("model_max_length:", model_max_length)
    model_max_length = ((model_max_length - 1) // 32 + 1) * 32
    tokenizer.save_pretrained(tokenizer_path, model_max_length=model_max_length)



def make_dataset(function_list, score_list, tokenizer_path):
    dataset_path = 'dataset'
    c_1 = 100

    tokenizer = Tokenizer.from_pretrained(tokenizer_path)
    tokens_list = tokenizer(function_list)
    print('tokenizer done')

    score_list = np.array(score_list)
    if np.max(score_list) == np.min(score_list):
        raise Exception('todo')
    score_list = (score_list - np.min(score_list)) / (np.max(score_list) - np.min(score_list))
    score_list = np.exp(c_1 * (score_list - 1))

    data_list = []
    for idx, ((input_ids, labels), score) in enumerate(zip(tokens_list, score_list)):
        print(idx, len(tokens_list), end='           \r')
        data = dict(input_ids=input_ids, labels=labels, score=score)
        data_list.append(data)

    dataset = Dataset.from_list(data_list)
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.save_to_disk(dataset_path)



def main():
    files = []
    files.extend(glob.glob('../zy1/funsearch_llm_api/samples/*.json'))
    files.extend(glob.glob('../zy2/funsearch_llm_api/samples/*.json'))
    files = natsort.natsorted(files)

    function_set = set()
    decision_set = set()
    function_list = []
    score_list = []
    function_total_set = set()

    for file in files:
        with open(file, 'r') as f:
            info = json.load(f)
        sample_order = info['sample_order']
        function = info['function']
        score = info['score']
        decisions = info['decisions']

        function_total_set.add(function)
        if score:
            matches = list(re.finditer(sample_iterator.SAMPLE_REGULAR, function))
            for match in reversed(matches):
                options = match.group(1).split(sample_iterator.SPLIT_CHAR)
                options = [x.strip() for x in options]
                decision_set.update(options)

            function_set.add(function)
            # decision_set.update(decisions)

            if len(matches) != len(decisions):
                raise Exception('matches len not equal to decision len')

            function_list.append((function, decisions))
            score_list.append(score)

    print(len(function_total_set), len(function_set), len(files), len(function_list))

    vocabulary = set()
    for function in function_set:
        function = re.sub(sample_iterator.SAMPLE_REGULAR, '', function)
        tokens = re.findall(FINDALL_RE, function)
        vocabulary.update(tokens)

    vocabulary.update(list(SPLIT_CHARS))
    vocabulary.update(decision_set)
    vocab_list = list(vocabulary)
    vocab_list.sort()
    for token in SPECIAL_TOKENS:
        if token in vocab_list:
            vocab_list.remove(token)
    vocab_list = SPECIAL_TOKENS + vocab_list
    print(len(vocab_list))

    tokenizer_path = 'tokenizer'
    tokenizer = Tokenizer()
    tokenizer.save_pretrained(tokenizer_path, vocab_list=vocab_list)

    # # test_model_max_length
    test_model_max_length(function_list, tokenizer_path)

    make_dataset(function_list, score_list, tokenizer_path)
    



if __name__ == '__main__':
    main()