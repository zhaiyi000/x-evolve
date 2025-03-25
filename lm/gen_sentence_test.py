import glob
import natsort
import json


def test1():
    files = []
    files.extend(glob.glob('../zy1/funsearch_llm_api/samples/*.json'))
    files.extend(glob.glob('../zy2/funsearch_llm_api/samples/*.json'))
    files = natsort.natsorted(files)

    function_dic = {}
    max_score = -1e10
    visited_decisions: dict[str, dict[tuple[str], float]] = {}

    for file in files:
        with open(file, 'r') as f:
            info = json.load(f)
        sample_order = info['sample_order']
        function = info['function']
        score = info['score']
        decisions = info['decisions']

        if function not in visited_decisions:
            visited_decisions[function] = {}
        visited_decisions[function][tuple(decisions)] = score

        if score:
            max_score = max(max_score, score)
            if function not in function_dic:
                function_dic[function] = score
            else:
                function_dic[function] = max(score, function_dic[function])

    print('len function_dic', len(function_dic))
    score_list = []
    visit_list = []
    length_list = []
    function_list = []
    idx = 0
    for function, score in function_dic.items():
        score_list.append(score)
        visit_list.append(0)
        length_list.append(len(function))
        function_list.append(function)
        
        with open(f'gen/{idx}.json', 'w') as f:
            f.write(json.dumps(function))
            f.write('\n')
            dic_score_list = list(visited_decisions[function].items())
            dic_score_list.sort(key=lambda x: x[1] if x[1] else -1e10, reverse=True)
            for dicisions, score in dic_score_list:
                f.write(str(dicisions))
                f.write('\t')
                f.write(str(score))
                f.write('\n')

        idx += 1

    return score_list, visit_list, length_list, function_list, max_score, visited_decisions


test1()