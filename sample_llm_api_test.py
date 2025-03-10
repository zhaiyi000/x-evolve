# from sample_llm_api import LLM
from sample_llm_api import EvaluateLLM
from sample_llm_api import llm_list




llm = EvaluateLLM(llm_list, 0.0, 5.0)
'''
llm.call_llm(llm_list[0], 10.0)
p = llm.calculate_probability()
print(p)
'''
'''
llm_index = np.random.choice(len(llm_list))
llm = llm_list[llm_index]
while:
    score = funsearch(llm)
    llm = llm.call_llm(llm, score)
'''
first = llm.calculate_probability()
print(first.llm_name)
next = llm.call_llm(first, 10.0)
print(next.llm_name)
