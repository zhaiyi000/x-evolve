import dataclasses
import numpy as np
from implementation import sample_iterator
import math
byte_key = "Bearer f184bcd9-68b0-49be-8a3f-ea095ee71e14"
open_key = "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"
rate = 7.2739
'''
llm_name: 大模型的名字
api_key: 调用大模型的密钥
Model: 大模型的型号
provider: 大模型的提供商
input_price: 输入的价格,RMB/million tokens
output_price: 输出的价格,RMB/million tokens
response_score: 记录大模型回复的得分
score: 大模型的评分
'''
@dataclasses.dataclass
class LLM:
    llm_name: str
    api_key: str
    model: str
    provider: str
    input_price: float
    output_price: float

llm_list = [
    LLM(
        llm_name="DeepSeek-R1",
        api_key=byte_key,
        model="ep-20250303202036-j6hfh",
        provider=None,
        input_price=4.0,
        output_price=16.0,
    ),
    LLM(
        llm_name="DeepSeek-V3", 
        api_key=byte_key, 
        model="ep-20250227102412-tfkv8", 
        provider=None, 
        input_price=2.0,
        output_price=8.0, 
    ),
    LLM(
        llm_name="Deepseek-R1-distill-qwen-7b",
        api_key=byte_key,
        model="ep-20250305162451-qxgns",
        provider=None,
        input_price=0.6,
        output_price=2.4,
    ),
    LLM(
        llm_name="DeepSeek-R1-distill-qwen-32b",
        api_key=byte_key,
        model="ep-20250305162537-2sbpv",
        provider=None,
        input_price=1.5,
        output_price=6.0,
    ),
    LLM(
        llm_name="GPT-4o",
        api_key=open_key,
        model=None,
        provider="OpenAI",
        input_price=2.5 * rate,
        output_price=10 * rate,
    ),
    LLM(
        llm_name="GPT-4o-mini",
        api_key=open_key,
        model=None,
        provider="OpenAI",
        input_price=0.15 * rate,
        output_price=0.6 * rate,
    ),
#    LLM(
#        llm_name="GPT-o1",
#        api_key=open_key,
#        model=None,
#        provider="OpenAI",
#        input_price=15 * rate,
#        output_price=60 * rate,
#    ),
    LLM(
        llm_name="GPT-o1-mini",
        api_key=open_key,
        model=None,
        provider="OpenAI",
        input_price=1.1 * rate,
        output_price=4.4 * rate,
    ),
    LLM(
        llm_name="GPT-o3-mini",
        api_key=open_key,
        model=None,
        provider="OpenAI",
        input_price=1.1 * rate,
        output_price=4.4 * rate,
    ),
    LLM(
        llm_name="GPT-o3-mini-high",
        api_key=open_key,
        model=None,
        provider="OpenAI",
        input_price=1.1 * rate,
        output_price=4.4 * rate,
    ),
    LLM(
        llm_name="Claude-3.7-sonnet",
        api_key=open_key,
        model=None,
        provider="Anthropic",
        input_price=3 * rate,
        output_price=15 * rate,
    ),
    LLM(
        llm_name="Gemini-2.0-flash-001",
        api_key=open_key,
        model=None,
        provider="Google AI Studio",
        input_price=0.1 * rate,
        output_price=0.4 * rate,
    ),
    LLM(
        llm_name="Gemini-2.0-pro-exp-02-05",
        api_key=open_key,
        model=None,
        provider="Google AI Studio",
        input_price=0.0 * rate,
        output_price=0.0 * rate,
    ),
    LLM(
        llm_name="Gemini-2.0-flash-thinking-exp",
        api_key=open_key,
        model=None,
        provider="Google AI Studio",
        input_price=0.0 * rate,
        output_price=0.0 * rate,
    ),
    LLM(
        llm_name="Grok-beta",
        api_key=open_key,
        model=None,
        provider="xAI",
        input_price=5 * rate,
        output_price=15 * rate,
    ),
]
'''
llm_list: 存储所有的大模型的list[LLM]
basescore: response的基准评分
self._max_score: response的最高得分,初始为负无穷
base_price_score: 价格评分的基准值
'''
# llm_sample_file = 'llm_sample.pkl'
# llm_sample_list = []
# import pickle
class EvaluateLLM:

    def __init__(self):
        global llm_list
        min_input_price = min([x.input_price for x in llm_list if x.input_price != 0])
        min_output_price = min([x.output_price for x in llm_list if x.output_price != 0])
        for llm in llm_list:
            if llm.input_price == 0:
                llm.input_price = min_input_price
            if llm.output_price == 0:
                llm.output_price = min_output_price
        self._llm_list = llm_list
        self._response_score = [[] for _ in llm_list]
    

    # 每次调用大模型后,将得分记录到大模型的Response_score中
    def call_llm(self, llm: LLM, parent_score: list[float], score: float):
        score = max(-500, score)
        self._response_score[self._llm_list.index(llm)].append((parent_score, score))
        

    # 计算下次选择大模型时使用的评分概率
    def calculate_probability(self) -> LLM:
        window_size = 1
        c_1 = 100
        temperature = 1

        score_list = []
        for llm, response_score in zip(self._llm_list, self._response_score):
            if len(response_score) < window_size:
                pass
            else:
                for par_score, child_score in response_score[-window_size:]:
                    score_list.append(child_score)

        if len(score_list) == 0:
            index = np.random.choice(len(self._llm_list))
        else:
            max_score = max(score_list)
            min_score = min(score_list)
            def cal_intensity(score):
                if max_score == min_score:
                    score = 1
                else:
                    score = (score - min_score) / (max_score - min_score)
                intensity = math.exp(c_1 * (score - 1))
                return intensity
                
            benefit_list = []
            for llm, response_score in zip(self._llm_list, self._response_score):
                if len(response_score) < window_size:
                    benefit_list.append(None)
                else:
                    benefit = 0
                    for par_score, child_score in response_score[-window_size:]:
                        intensity = cal_intensity(child_score)
                        for par_s in par_score:
                            benefit += intensity * (child_score - par_s)
                    benefit_list.append(benefit / (llm.input_price + llm.output_price))

            max_benefit = max([x for x in benefit_list if x])
            benefit_list = [x if x is not None else max_benefit for x in benefit_list]
            probabilities = sample_iterator.softmax(benefit_list, temperature)
            index = np.random.choice(len(self._llm_list), p=probabilities)
        
        # llm_sample_list.append((self._response_score, index))
        # with open(llm_sample_file, 'wb') as f:
        #     pickle.dump(llm_sample_list, f)
        return self._llm_list[index]


def get_qwen_32b() -> LLM:
    llm_name = "DeepSeek-R1-distill-qwen-32b"
    for llm in llm_list:
        if llm.llm_name == llm_name:
            return llm
    raise ValueError(f"LLM {llm_name} not found")
    
