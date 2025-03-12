import glob
import json
import re
import natsort
import black
import multiprocessing
import tokenize
from io import BytesIO
import ast
import math
import numpy as np


def remove_unassigned_strings(node):
    """
    移除 AST 中未赋值给变量的字符串常量节点。
    """
    # 如果节点有 body 属性（如函数、类或模块）
    if hasattr(node, 'body') and isinstance(node.body, list):
        new_body = []
        for child in node.body:
            # 检查是否是未赋值的字符串常量
            if not (isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant) and isinstance(child.value.value, str)):
                new_body.append(child)
        node.body = new_body

    # 递归检查子节点
    for child in ast.iter_child_nodes(node):
        remove_unassigned_strings(child)


def remove_comments(code):
    # 用于存储去掉注释后的代码
    result = []
    # 将字符串编码为字节流，供 tokenize 使用
    tokens = tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)
    
    for toknum, tokval, _, _, _ in tokens:
        # 忽略注释和编码标记
        if toknum != tokenize.COMMENT and toknum != tokenize.NL:
            result.append((toknum, tokval))
    
    # 将代码片段组合为字符串
    code_without_comments = tokenize.untokenize(result).decode('utf-8')

    # 解析为 AST
    tree = ast.parse(code_without_comments)
    # 移除 AST 中的 docstring
    remove_unassigned_strings(tree)
    # 还原为代码
    code_without_docstrings = ast.unparse(tree)
    return code_without_docstrings


def remove_raise(code_str):
    # # 去掉多行注释（/* ... */）
    # code_str = re.sub(r'""".*?"""|\'\'\'.*?\'\'\'', '', code_str, flags=re.DOTALL)
    # # 去掉单行注释（# ...）
    # code_str = re.sub(r'#.*', '', code_str)

    code_str = re.sub(r'raise.*', 'raise Exception()', code_str)

    return code_str


def handle_file(file_i: int, file: str):
    # print(file_i)
    with open(file, 'r') as f:
        info = json.load(f)
    sample_order = info['sample_order']
    function = info['function']
    score = info['score']
    if score == None or 'return' not in function:
        # print('.', end='', flush=True)
        # print('score is None', file)
        return
    
    function = remove_comments(function)
    function = remove_raise(function)

    try:
        function = black.format_str(function, mode=black.Mode())
        function = re.sub(r'(\S)\*\*(\S)', r'\1 ** \2', function)
        function = re.sub(r'-(?!>| |=)', r'- ', function)
    except:
        # import pdb; pdb.set_trace()
        print('*', end='', flush=True)
        return

    header = 'def priority(item: float, bins: np.ndarray) -> np.ndarray:'

    if header not in function:
        print('header not in function')
        return
    body = function.replace(header, '')
    body = body.replace('    ', '\t')

    if score < -500 or score > -207.70:
        print('error')
        raise Exception('todo')
    
    def reward_func(x):
        up = -207.70
        return np.exp(x - up)
    
    return dict(text=body, score=score, rewards=reward_func(score), sample_order=sample_order) if body else None


def main():

    files = []
    files.extend(glob.glob('samples/*.json'))
    files.extend(glob.glob('samples_2/*.json'))
    files = natsort.natsorted(files)
    print(len(files))


    results = []
    with multiprocessing.Pool() as pool:
        for file_i, file in enumerate(files):
            # handle_file(file_i, file)
            result = pool.apply_async(handle_file, args=(file_i, file))
            results.append(result)
        pool.close()
        pool.join()


    body_list = []
    for result in results:
        if result.get() is not None:
            body_list.append(result.get())

    print('len body_list', len(body_list))
    text_set = set()
    with open('json/body.json', 'w') as f:
        for body in body_list:
            if body['text'] not in text_set:
                text_set.add(body['text'])
                json.dump(body, f)
                f.write('\n')


if __name__ == '__main__':
    main()