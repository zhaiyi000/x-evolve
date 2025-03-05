import dataclasses
import numpy as np
from implementation import sample_iterator
'''
LLM_Name: 大模型的名字
API_Key: 调用大模型的密钥
Model: 大模型的型号
Provider: 大模型的提供商
Input_price: 输入的价格,RMB/million tokens
Output_price: 输出的价格,RMB/million tokens
'''
@dataclasses.dataclass
class LLM:
    LLM_Name: str
    API_Key: str
    Model: str
    Provider: str
    Input_price: float
    Output_price: float
    Response_score: list = dataclasses.field(default_factory=list)
    Score: float = 0

'''
llm_list: 存储所有的大模型的list[LLM]
basescore: 初始评分
self._Max_score: 最高得分,初始为负无穷
'''
class Evaluate_LLM:

    def __init__(self, llm_list: list[LLM], basescore: float):
        self._llm_list = llm_list
        self._Max_score = float("-inf")
        self._basescore = basescore
        for index in range(len(self._llm_list)):
            self._llm_list[index].Score = -self._llm_list[index].Input_price - self._llm_list[index].Output_price

    # 将大模型注册到评分器中
    def _register_llm(self, llm: LLM):
        self._llm_list.append(llm)
        index = self._llm_list.index(llm)
        # 根据价格初始化初始评分
        init_score = -llm.Input_price - llm.Output_price
        self._llm_list[index].Score = init_score
    
    # 每次调用大模型后,将得分记录到大模型的Response_score中
    def call_llm(self, llm: LLM, score: float):
        index = self._llm_list.index(llm)
        if(self._Max_score == float("-inf")):
            self._Max_score = score
        self._llm_list[index].Response_score.append((score-self._basescore, score-self._Max_score))
        self._Max_score = max(self._Max_score, score)
        if len(llm.Response_score) == 1:
            '''
            该模型只被调用过一次
            score1: 根据该次调用的得分与basescore的差值对模型进行评分
            score2: 根据该次调用的得分与Max_score的差值对模型进行评分
            '''
            score1 = self._llm_list[index].Response_score[0][0]
            score2 = self._llm_list[index].Response_score[0][1]
            self._llm_list[index].Score = self._llm_list[index].Score + score1 + score2
        else:
            '''
            该模型被调用过多次
            score1: 根据该次调用的得分与basescore的差值对模型进行评分
            score2: 根据该次调用的得分与Max_score的差值对模型进行评分
            score3: 根据该次调用的得分与上一次调用的得分的差值对模型进行评分
            '''
            score1 = self._llm_list[index].Response_score[-1][0]
            score2 = self._llm_list[index].Response_score[-1][1]
            score3 = self._llm_list[index].Response_score[-1][0] - self._llm_list[index].Response_score[-2][0]
            self._llm_list[index].Score = self._llm_list[index].Score + score1 + score2 + score3

    # 计算下次选择大模型时使用的评分概率
    def calculate_probability(self):
        score_list = np.array([llm.Score for llm in self._llm_list])
        print(score_list)
        temperature = 1.0
        probability = sample_iterator.softmax(score_list, temperature)
        return probability


