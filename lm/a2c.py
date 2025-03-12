import gym
from gym import spaces
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from stable_baselines3 import A2C, Reinforce
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

class MyPolicy:
    def __init__(self, *args, **kwds):
        pass

    def __call__(self, *args, **kwds):
        pass

    def to(self, device):
        pass


class GPT2TextGenEnv(gym.Env):
    """
    自定义环境：基于 GPT-2 的文本生成。
    """
    def __init__(self, model, tokenizer, max_len=50):
        super(GPT2TextGenEnv, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 动作空间：GPT-2的词汇表大小
        self.action_space = spaces.Discrete(len(tokenizer))
        
        # 状态空间：生成的 token 序列
        self.observation_space = spaces.Box(
            low=0, high=len(tokenizer), shape=(max_len,), dtype=np.int32
        )

        self.reset()

    def reset(self):
        """
        重置环境：初始化状态为空。
        """
        self.tokens = []  # 存储当前生成的 token 序列
        self.text = ""  # 当前生成的文本
        return [(np.array([1], dtype=np.int64), np.array([1], dtype=np.int64))]

    def step(self, action):
        """
        执行动作：更新生成的 token，并计算奖励。
        """
        self.tokens.append(action)
        input_ids = torch.tensor([self.tokens]).to(self.model.device)
        
        # 用 GPT-2 模型预测下一个 token 的概率分布
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]  # 取最后一个 token 的 logits

        # 转换 logits 为概率分布
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
        
        # 更新当前生成文本
        self.text = self.tokenizer.decode(self.tokens, skip_special_tokens=True)

        # 奖励计算：基于生成文本和模型输出概率
        reward = self.get_reward(self.text, probabilities)
        
        # 判断是否达到终止条件
        done = len(self.tokens) >= self.max_len
        next_state = np.zeros(self.max_len, dtype=np.int32)
        next_state[:len(self.tokens)] = self.tokens
        
        return next_state, reward, done, {}

    def get_reward(self, text, probabilities):
        """
        自定义奖励函数：
        - 奖励包括生成文本的长度、多样性奖励和模型预测的置信度。
        """
        length_reward = len(text) / self.max_len
        diversity_reward = len(set(text.split())) / max(len(text.split()), 1)
        confidence_reward = np.mean(probabilities)  # 概率的均值
        
        # 加权组合奖励
        total_reward = 0.5 * length_reward + 0.3 * diversity_reward + 0.2 * confidence_reward
        return total_reward

    def render(self, mode='human'):
        """
        可视化生成文本。
        """
        print(self.text)


# class MyEnv:
#     pass


# 加载 GPT-2 模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
# model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# 创建自定义环境
# env = DummyVecEnv([lambda: GPT2TextGenEnv(model, tokenizer)])
env = GPT2TextGenEnv(model, tokenizer)

# A2C 超参数
a2c_params = {
    "policy": MyPolicy,
    "env": env,
    "learning_rate": 1e-4
}

# 创建 A2C 模型
a2c_model = A2C(**a2c_params)

# 自定义回调函数，用于监控训练进度
class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        """
        每步执行的回调：打印平均奖励。
        """
        if self.n_calls % 100 == 0:
            print(f"Step: {self.n_calls}, Reward: {np.mean(self.locals['rewards'])}")
        return True

# 开始训练
callback = CustomCallback()
a2c_model.learn(total_timesteps=5000, callback=callback)

# 保存模型
a2c_model.save("a2c_gpt2")

# 测试生成文本
def generate_text(input_text, model, tokenizer, max_len=50):
    """
    用 GPT-2 生成文本。
    """
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    output = model.generate(input_ids, max_length=max_len, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 测试生成文本
input_text = "In a distant future"
generated_text = generate_text(input_text, model, tokenizer)
print("Generated Text:", generated_text)