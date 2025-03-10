from sample_llm_api import EvaluateLLM
from sample_llm_api import get_qwen_32b
import numpy as np




llm = EvaluateLLM()
llm_iterator = llm.calculate_probability()
print(llm_iterator.llm_name)
for i in range(0, 10):
    score = np.random.randint(-500, -210) # score = funsearch(llm_iterator)
    llm_iterator = llm.call_llm(llm_iterator, score)
    print(llm_iterator.llm_name)

qwen = get_qwen_32b()
print(qwen.llm_name)
