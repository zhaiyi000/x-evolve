import dataclasses
import numpy as np
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
    response_score: list = dataclasses.field(default_factory=list)
    score: float = 0

'''
llm_list: 存储所有的大模型的list[LLM]
basescore: 基准评分
self._Max_score: 最高得分,初始为负无穷
Base_price_score: 价格评分的基准值
'''
class EvaluateLLM:

    def __init__(self, llm_list: list[LLM], basescore: float, Base_price_score: float):
        self._llm_list = llm_list
        self._Max_score = float("-inf")
        self._basescore = basescore
        self._Base_price_score = Base_price_score
        # 根据价格初始化初始评分
        self._total_cost = np.sum([llm.input_price + llm.output_price for llm in self._llm_list])
        for index in range(len(self._llm_list)):
            # 计算价格权重占比
            cost_ratio = (self._llm_list[index].input_price + self._llm_list[index].output_price) / self._total_cost
            self._llm_list[index].score = self._Base_price_score * (1 - cost_ratio)

    # 将大模型注册到评分器中
    def _register_llm(self, llm: LLM):
        self._llm_list.append(llm)
        index = self._llm_list.index(llm)
        # 根据价格初始化初始评分
        self._total_cost = self._total_cost + llm.input_price + llm.output_price
        # 计算价格权重占比
        cost_ratio = (llm.input_price + llm.output_price) / self._total_cost
        self._llm_list[index].score = self._Base_price_score * (1 - cost_ratio)
    
    # 每次调用大模型后,将得分记录到大模型的Response_score中
    def call_llm(self, llm: LLM, score: float):
        index = self._llm_list.index(llm)
        if(self._Max_score == float("-inf")):
            self._Max_score = score
        self._llm_list[index].response_score.append((score-self._basescore, score-self._Max_score))
        self._Max_score = max(self._Max_score, score)

        decay_factor = 0.8 # 历史衰减系数
        if len(llm.response_score) == 1:
            '''
            该模型只被调用过一次
            score1: 根据该次调用的得分与basescore的差值对模型进行评分
            score2: 根据该次调用的得分与Max_score的差值对模型进行评分
            '''
            score1 = self._llm_list[index].response_score[0][0]
            score2 = self._llm_list[index].response_score[0][1]
            delta = (score1 + score2)
        else:
            '''
            该模型被调用过多次
            score1: 根据该次调用的得分与basescore的差值对模型进行评分
            score2: 根据该次调用的得分与Max_score的差值对模型进行评分
            score3: 根据该次调用的得分与上一次调用的得分的差值对模型进行评分
            '''
            score1 = self._llm_list[index].response_score[-1][0]
            score2 = self._llm_list[index].response_score[-1][1]
            score3 = self._llm_list[index].response_score[-1][0] - self._llm_list[index].response_score[-2][0]
            # 模型调用次数越多,当前次数的权重越大
            delta = (score1 + score2 + score3) * (1 - decay_factor ** len(llm.response_score))
        # 更新模型的评分(平滑过渡)
        self._llm_list[index].score = decay_factor * self._llm_list[index].score + (1 - decay_factor) * delta

    # 计算下次选择大模型时使用的评分概率
    def calculate_probability(self):
        score_list = np.array([llm.score for llm in self._llm_list])
        # 标准化得分
        scores = (score_list - np.min(score_list)) / (np.max(score_list) - np.min(score_list) + 1e-6)
        # 动态温度调整（鼓励探索）
        min_temperature = 0.5 # 最小温度
        temperature = max(min_temperature, 1.0 - 0.1 * len(self._llm_list[0].response_score))
        # 带平滑的softmax（确保最小概率）
        exp_scores = np.exp(scores / temperature)
        probabilities = exp_scores / (exp_scores.sum() + 1e-6)
        ratio = 0.9 # 混合均匀分布的比例
        probabilities = ratio * probabilities + (1 - ratio) * (1/len(probabilities))  # 混合均匀分布
        return probabilities


