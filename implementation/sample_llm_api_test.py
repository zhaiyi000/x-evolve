from sample_llm_api import EvaluateLLM
from sample_llm_api import get_qwen_32b
import numpy as np




def test1():
    llm = EvaluateLLM()
    llm_iterator = llm.calculate_probability()
    print(llm_iterator.llm_name)
    for i in range(0, 10):
        score = np.random.randint(-500, -210) # score = funsearch(llm_iterator)
        llm_iterator = llm.call_llm(llm_iterator, score)
        print(llm_iterator.llm_name)

    qwen = get_qwen_32b()
    print(qwen.llm_name)




def test2():
    import pickle
    llm_sample_file = 'llm_sample_bak.pkl'
    with open(llm_sample_file, 'rb') as f:
        llm_sample_list = pickle.load(f)

    llm = EvaluateLLM()
    llm._response_score = llm_sample_list[-1][0]
    llm.calculate_probability()


test2()

