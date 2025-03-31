import dataclasses
import numpy as np
from implementation import sample_iterator
import math
import copy
from config import log_dir
byte_key = "Bearer f184bcd9-68b0-49be-8a3f-ea095ee71e14"
open_key = "Bearer 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"
byte_http = 'https://ark.cn-beijing.volces.com/api/v3/chat/completions'
open_http = 'https://openrouter.ai/api/v1/chat/completions'
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
    request_http: str
    input_price: float
    output_price: float

llm_list = [
    LLM(
        llm_name="DeepSeek-R1",
        api_key=byte_key,
        model="ep-20250303202036-j6hfh",
        provider=None,
        request_http = byte_http,
        input_price=4.0,
        output_price=16.0,
    ),
    LLM(
        llm_name="DeepSeek-V3", 
        api_key=byte_key, 
        model="ep-20250331155118-wgw4n", 
        provider=None, 
        request_http = byte_http,
        input_price=2.0,
        output_price=8.0, 
    ),
    # LLM(
    #     llm_name="Deepseek-R1-distill-qwen-7b",
    #     api_key=byte_key,
    #     model="ep-20250305162451-qxgns",
    #     provider=None,
    #     request_http = byte_http,
    #     input_price=0.6,
    #     output_price=2.4,
    # ),
    LLM(
        llm_name="DeepSeek-R1-distill-qwen-32b",
        api_key=byte_key,
        model="ep-20250305162537-2sbpv",
        provider=None,
        request_http = byte_http,
        input_price=1.5,
        output_price=6.0,
    ),
    # LLM(
    #     llm_name="GPT-4o",
    #     api_key=open_key,
    #     model="openai/gpt-4o-2024-11-20",
    #     provider="OpenAI",
    #     request_http = open_http,
    #     input_price=2.5 * rate,
    #     output_price=10 * rate,
    # ),
    LLM(
        llm_name="GPT-4o-mini",
        api_key=open_key,
        model="openai/gpt-4o-mini",
        provider="OpenAI",
        request_http = open_http,
        input_price=0.15 * rate,
        output_price=0.6 * rate,
    ),
    # not find core code
    # LLM(
    #     llm_name="GPT-o1",
    #     api_key=open_key,
    #     model="openai/o1",
    #     provider="OpenAI",
    #     request_http = open_http,
    #     input_price=15 * rate,
    #     output_price=60 * rate,
    # ),
    # LLM(
    #     llm_name="GPT-o1-mini",
    #     api_key=open_key,
    #     model="openai/o1-mini",
    #     provider="OpenAI",
    #     request_http = open_http,
    #     input_price=1.1 * rate,
    #     output_price=4.4 * rate,
    # ),
    # not find core code
    # LLM(
    #     llm_name="GPT-o3-mini",
    #     api_key=open_key,
    #     model="openai/o3-mini",
    #     provider="OpenAI",
    #     request_http = open_http,
    #     input_price=1.1 * rate,
    #     output_price=4.4 * rate,
    # ),
    # not find core code
    # LLM(
    #     llm_name="GPT-o3-mini-high",
    #     api_key=open_key,
    #     model="openai/o3-mini-high",
    #     provider="OpenAI",
    #     request_http = open_http,
    #     input_price=1.1 * rate,
    #     output_price=4.4 * rate,
    # ),
    # LLM(
    #     llm_name="Claude-3.7-sonnet",
    #     api_key=open_key,
    #     model="anthropic/claude-3.7-sonnet",
    #     provider="Anthropic",
    #     request_http = open_http,
    #     input_price=3 * rate,
    #     output_price=15 * rate,
    # ),
    # LLM(
    #     llm_name="Claude-3.7-Sonnet-thinking",
    #     api_key=open_key,
    #     model="anthropic/claude-3.7-sonnet:thinking",
    #     provider="Google",
    #     request_http = open_http,
    #     input_price=3 * rate,
    #     output_price=15 * rate,
    # ),
    LLM(
        llm_name="Gemini-2.0-flash-001",
        api_key=open_key,
        model="google/gemini-2.0-flash-001",
        provider="Google AI Studio",
        request_http = open_http,
        input_price=0.1 * rate,
        output_price=0.4 * rate,
    ),
    # LLM(
    #     llm_name="Gemini-2.0-pro-exp-02-05",
    #     api_key=open_key,
    #     model="google/gemini-2.0-pro-exp-02-05:free",
    #     provider="Google AI Studio",
    #     request_http = open_http,
    #     input_price=0.0 * rate,
    #     output_price=0.0 * rate,
    # ),
    # LLM(
    #     llm_name="Gemini-2.0-flash-thinking-exp",
    #     api_key=open_key,
    #     model="google/gemini-2.0-flash-thinking-exp:free",
    #     provider="Google AI Studio",
    #     request_http = open_http,
    #     input_price=0.0 * rate,
    #     output_price=0.0 * rate,
    # ),
    # LLM(
    #     llm_name="Grok-beta",
    #     api_key=open_key,
    #     model="x-ai/grok-beta",
    #     provider="xAI",
    #     request_http = open_http,
    #     input_price=5 * rate,
    #     output_price=15 * rate,
    # ),

    LLM(
        llm_name="Qwen2.5-72B-Instruct",
        api_key=open_key,
        model="qwen/qwen-2.5-72b-instruct",
        provider="Nebius",
        request_http = open_http,
        input_price=0.13 * rate,
        output_price=0.4 * rate,
    ),
]
'''
llm_list: 存储所有的大模型的list[LLM]
basescore: response的基准评分
self._max_score: response的最高得分,初始为负无穷
base_price_score: 价格评分的基准值
'''
llm_sample_list = []
import pickle
import os
class EvaluateLLM:

    def __init__(self):
        global llm_list
        # min_input_price = min([x.input_price for x in llm_list if x.input_price != 0])
        # min_output_price = min([x.output_price for x in llm_list if x.output_price != 0])
        # for llm in llm_list:
        #     if llm.input_price == 0:
        #         llm.input_price = min_input_price
        #     if llm.output_price == 0:
        #         llm.output_price = min_output_price
        self._llm_list = llm_list
        self._response_score = [[] for _ in llm_list]
        total_price = sum(llm.input_price + llm.output_price for llm in llm_list)
        self._price_score = [(1-(llm.input_price+llm.output_price)/total_price) for llm in llm_list]
    

    # 每次调用大模型后,将得分记录到大模型的Response_score中
    def call_llm(self, llm: LLM, parent_score: list[float], score: float):
        score = max(-500, score)
        self._response_score[self._llm_list.index(llm)].append((parent_score, score))
        

    # 计算下次选择大模型时使用的评分概率
    def calculate_probability(self) -> LLM:
        window_size = 5  # window_size是否过大了，对历史的观察太多 todo
        c_1 = 100
        temperature = 1
        decay = 0.8

        score_list = []
        probabilities = []
        benefit_list = []
        benefit_list_origin = []
        for llm, response_score in zip(self._llm_list, self._response_score):
            if len(response_score) < window_size:
                pass
            else:
                for par_score, child_score in response_score[-window_size:]:
                    score_list.append(child_score)

        if len(score_list) == 0:
            probabilities = [1/len(llm_list) for _ in llm_list]
            index = np.random.choice(len(self._llm_list))
            # print('-----------random select------------')
        else:
            max_score = max(score_list)
            min_score = min(score_list)
            def cal_intensity(score):
                if max_score == min_score:
                    score = 1
                else:
                    score = (score - min_score) / (max_score - min_score)
                intensity = math.exp(c_1 * (score - 1))
                # intensity = score
                return intensity
                
            benefit_list = []
            for llm, response_score, price_score in zip(self._llm_list, self._response_score, self._price_score):
                if len(response_score) < window_size:
                    benefit_list_origin.append(0)
                    benefit_list.append(None)
                else:
                    benefit = 0
                    his_num = 0
                    for par_score, child_score in response_score[-window_size:]:
                        his_num += 1
                        intensity = cal_intensity(child_score)
                        for par_s in par_score:
                            benefit += intensity * (child_score - par_s)
                            # *(decay ** his_num)
                    benefit_list_origin.append(benefit)
                    benefit_list.append(benefit / (llm.input_price + llm.output_price))

            max_benefit = max([x for x in benefit_list if x])
            benefit_list = [x if x is not None else max_benefit for x in benefit_list]
            probabilities = sample_iterator.softmax(benefit_list, temperature)
            index = np.random.choice(len(self._llm_list), p=probabilities)
        # print('------------response score list--------------')
        # print(self._response_score)
        # print(benefit_list)
        # print(probabilities)
        # print(index)
        # print('---------------------------------------------')
        res = copy.deepcopy(self._response_score)
        llm_sample_list.append((res, benefit_list_origin, benefit_list, probabilities, index))
        llm_sample_file = f'{log_dir}/llm_sample.pkl'
        with open(llm_sample_file, 'wb') as f:
            pickle.dump(llm_sample_list, f)
        return self._llm_list[index]


def get_qwen_32b() -> LLM:
    llm_name = "DeepSeek-R1-distill-qwen-32b"
    for llm in llm_list:
        if llm.llm_name == llm_name:
            return llm
    raise ValueError(f"LLM {llm_name} not found")

def get_llm(llm_name: str) -> LLM:
    for llm in llm_list:
        if llm.llm_name == llm_name:
            return llm
    raise ValueError(f"LLM {llm_name} not found")
    



def get_deepseek_v3() -> LLM:
    llm_name = "DeepSeek-V3"
    for llm in llm_list:
        if llm.llm_name == llm_name:
            return llm
    raise ValueError(f"LLM {llm_name} not found")
    

def get_claude_37_thinking() -> LLM:
    llm_name = "Claude-3.7-Sonnet-thinking"
    for llm in llm_list:
        if llm.llm_name == llm_name:
            return llm
    raise ValueError(f"LLM {llm_name} not found")


def get_random_model() -> LLM:
    return llm_list[np.random.choice(len(llm_list), size=1)[0]]
    