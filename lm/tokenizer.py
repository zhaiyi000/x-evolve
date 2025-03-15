
import json
import re
import multiprocessing
from datasets import load_dataset
import os
import glob
import natsort
from concurrent.futures import ProcessPoolExecutor

SPLIT_CHARS = '\{\}()[]\t\n: ,\'".+-=*/~|^?'
PAD_TOKEN = '[PAD]'
ANS_TOKEN = '[ANS]'
SEP_TOKEN = '[SEP]'
SPLIT_CHARS_RE = f'[{re.escape(SPLIT_CHARS)}]'
PAD_CHAR_RE = f'{re.escape(PAD_TOKEN)}'
ANS_CHAR_RE = f'{re.escape(ANS_TOKEN)}'
SEP_CHAR_RE = f'{re.escape(SEP_TOKEN)}'
FINDALL_RE = f'{PAD_CHAR_RE}|{ANS_CHAR_RE}|{SEP_CHAR_RE}|{SPLIT_CHARS_RE}|[^{SPLIT_CHARS_RE[1:-1]}]+'
SPECIAL_TOKENS = [PAD_TOKEN, ANS_TOKEN, SEP_TOKEN]


def tokenizer_encode_inner(vocab, pad_token_id, ans_token_id, sep_token_id, model_max_length, function, decision, idx):
    if idx % 1000 == 0:
        print(idx)
    words = re.findall(FINDALL_RE, function)
    ids = [vocab[word] for word in words]
    ids.append(ans_token_id)
    ids += [vocab[word] for word in decision]
    ids.append(sep_token_id)
    mask = [1] * len(ids)

    if model_max_length:
        ids += [pad_token_id] * (model_max_length-len(ids))
        mask += [0] * (model_max_length-len(mask))
    return (ids, mask)


def tokenizer_decode_inner(id_to_token, ids, err_queue, add_special_tokens, special_tokens):
    try:
        if add_special_tokens:
            return ''.join([id_to_token[x] for x in ids])
        else:
            return ''.join([id_to_token[x] for x in ids if x not in special_tokens])
    except Exception as err:
        if err_queue:
            err_queue.put(err)
        else:
            raise err


class Tokenizer():
    def __init__(self, dir_path=None):
        if dir_path:
            path = os.path.join(dir_path, 'tokenizer.json')
            with open(path, 'r') as f:
                tokenizer_dic = json.load(f)

            self.vocab = tokenizer_dic['vocab']
            
            id_to_token_dic = list(self.vocab.items())
            id_to_token_dic.sort(key=lambda x: x[1])
            self.id_to_token = [x[0] for x in id_to_token_dic]
            
            self._model_max_length = tokenizer_dic['model_max_length']
            self._pad_token_id = self.vocab[PAD_TOKEN]
            self._ans_token_id = self.vocab[ANS_TOKEN]
            self._sep_token_id = self.vocab[SEP_TOKEN]
            self.special_tokens = [self._pad_token_id, self._ans_token_id, self._sep_token_id]
            self.executor = None


    def encode(self, sentence, padding=None):
        model_max_length = self._model_max_length if padding == 'max_length' else None
        ids, mask = tokenizer_encode_inner(self.vocab, self._pad_token_id, self._cls_token_id, self._sep_token_id, model_max_length, sentence, None)
        return ids


    def batch_encode(self, function_list, decision_list, padding=None):
        model_max_length = self._model_max_length if padding == 'max_length' else None
        if len(function_list) == 0:
            raise Exception('todo')
        elif len(function_list) == 1:
            raise Exception('todo')
            # ids, mask = tokenizer_encode_inner(self.vocab, self._pad_token_id, self._cls_token_id, self._sep_token_id, model_max_length, sentence_list[0], None)
            # return dict(input_ids=[ids], attention_mask=[mask])
        else:
            if self.executor is None:
                self.executor = ProcessPoolExecutor(max_workers=min(os.cpu_count(), 64))

            executor = ProcessPoolExecutor(max_workers=min(os.cpu_count(), 64))
            futures = [executor.submit(tokenizer_encode_inner, self.vocab, self._pad_token_id, self._ans_token_id, self._sep_token_id, model_max_length, function, decision, idx) for idx, (function, decision) in enumerate(zip(function_list, decision_list))]
            result_list = []
            for future in futures:
                result = future.result()
                result_list.append(result)
            return result_list
    

    def decode(self, ids, add_special_tokens=True):
        return tokenizer_decode_inner(self.id_to_token, ids, None, add_special_tokens, self.special_tokens)
    

    def batch_decode(self, input_ids, add_special_tokens=True):
        if len(input_ids) == 0:
            return []
        # elif len(input_ids) == 1:
        else:
            sentence_list = []
            for ids in input_ids:
                sentence = tokenizer_decode_inner(self.id_to_token, ids, None, add_special_tokens, self.special_tokens)
                sentence_list.append(sentence)
            return sentence_list
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
    

    def __call__(self, function_list, decision_list, padding=None):
        return self.batch_encode(function_list, decision_list, padding=padding)


    @property
    def model_max_length(self):
        return self._model_max_length
    
    @property
    def pad_token_id(self):
        return self._pad_token_id
    
    # @property
    # def cls_token_id(self):
    #     return self._cls_token_id
    
    @property
    def sep_token_id(self):
        return self._sep_token_id

    @property
    def eos_token_id(self):
        return self._sep_token_id

    @property
    def vocab_size(self):
        return len(self.vocab)

    @classmethod
    def from_pretrained(cls, path):
        return Tokenizer(path)
    

    def save_pretrained(self, dir_path, vocab_list=None, model_max_length=None):
        if vocab_list is not None:
            vocab = {}
            for item_i, item in enumerate(vocab_list):
                vocab[item] = item_i
        elif hasattr(self, 'vocab'):
            vocab = self.vocab
        else:
            raise Exception('vocab is invalid')
        
        if model_max_length is None and hasattr(self, '_model_max_length'):
            model_max_length = self._model_max_length

        tokenizer_dic = dict(vocab=vocab, model_max_length=model_max_length)

        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, 'tokenizer.json')
        with open(path, 'w') as f:
            json.dump(tokenizer_dic, f, indent=2)


def test_model_max_length(function_list, decision_list, tokenizer_path):
    # data_files = {}
    # data_files["train"] = file
    # extension = data_files["train"].split(".")[-1]
    # raw_datasets = load_dataset(
    #     extension,
    #     data_files=data_files,
    #     keep_in_memory=True
    # )
    # raw_datasets["train"] = load_dataset(
    #     extension,
    #     data_files=data_files,
    #     split=f"train",
    #     keep_in_memory=True
    # )
    tokenizer = Tokenizer.from_pretrained(tokenizer_path)

    # def tokenize_function(examples):
    #     encode = tokenizer(examples["text"])
    #     input_ids = encode["input_ids"]
    #     nonlocal model_max_length
    #     for ids in input_ids:
    #         model_max_length = max(model_max_length, len(ids))
    #     return encode

    # tokenized_datasets = raw_datasets.map(
    #     tokenize_function,
    #     batched=True,
    #     num_proc=None,
    #     load_from_cache_file=True,
    #     desc="Running tokenizer on every text in dataset",
    #     keep_in_memory=True
    # )
    tokens_list = tokenizer(function_list, decision_list)

    model_max_length = max(len(tokens[0]) for tokens in tokens_list)

    print("model_max_length:", model_max_length)
    model_max_length = ((model_max_length - 1) // 32 + 1) * 32
    tokenizer.save_pretrained(tokenizer_path, model_max_length=model_max_length)


def main():
    files = []
    files.extend(glob.glob('samples_1/*.json'))
    files.extend(glob.glob('samples_2/*.json'))
    files.extend(glob.glob('samples_3/*.json'))
    files = natsort.natsorted(files)

    function_set = set()
    decision_set = set()
    function_list = []
    decision_list = []

    for file in files:
        with open(file, 'r') as f:
            info = json.load(f)
        sample_order = info['sample_order']
        function = info['function']
        score = info['score']
        decisions = info['decisions']

        if score:
            function_set.add(function)
            decision_set.update(decisions)

            function_list.append(function)
            decision_list.append(decisions)

    print(len(function_set), len(files), len(function_list))

    vocabulary = set()
    for function in function_set:
        tokens = re.findall(FINDALL_RE, function)
        vocabulary.update(tokens)

    vocabulary.update(list(SPLIT_CHARS))
    vocabulary.update(decision_set)

    # number_chars = '0123456789.'

    # with open('json/body.json', 'r') as f:
    #     body_list = f.read().strip().split('\n')
    #     body_list = [json.loads(x) for x in body_list]

    
    # for body in body_list:
    #     body = body['text']

    #     tokens = re.findall(FINDALL_RE, body)

    #     vocabulary.update(tokens)

    # vocabulary.update(list(SPLIT_CHARS))

    vocab_list = list(vocabulary)
    # vocab_list = [x for x in vocab_list if re.match(r'^-?\d+(\.\d+)?$', x) is None]
    # vocab_list.extend(list(number_chars))
    # vocab_list = list(set(vocab_list))
    vocab_list.sort()
    for token in SPECIAL_TOKENS:
        if token in vocab_list:
            vocab_list.remove(token)
    vocab_list = SPECIAL_TOKENS + vocab_list
    print(len(vocab_list))

    tokenizer_path = 'tokenizer'
    tokenizer = Tokenizer()
    tokenizer.save_pretrained(tokenizer_path, vocab_list=vocab_list)


    # test_model_max_length
    test_model_max_length(function_list, decision_list, tokenizer_path)
    



if __name__ == '__main__':
    main()