from sample_llm_api import LLM
from sample_llm_api import Evaluate_LLM

DeepSeek_R1 = LLM("DeepSeek-R1", "Bearer f184bcd9-68b0-49be-8a3f-ea095ee71e14", "ep-20250303202036-j6hfh", "", 4.0, 16.0, [])
DeepSeek_V3 = LLM("DeepSeek-V3", "Bearer f184bcd9-68b0-49be-8a3f-ea095ee71e14", "ep-20250227102412-tfkv8", "", 2.0, 8.0, [])
Deepseek_R1_distill_qwen_7b = LLM("Deepseek-R1-distill-qwen-7b", "Bearer f184bcd9-68b0-49be-8a3f-ea095ee71e14", "ep-20250305162451-qxgns", "", 0.6, 2.4, [])
DeepSeek_R1_distill_qwen_32b = LLM("DeepSeek-R1-distill-qwen-32b", "Bearer f184bcd9-68b0-49be-8a3f-ea095ee71e14", "ep-20250305162537-2sbpv", "", 1.5, 6.0, [])

GPT_4o = LLM("GPT-4o", "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", "", "OpenAI", 18.18475, 72.739, [])
GPT_4o_mini = LLM("GPT-4o-mini", "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", "", "OpenAI", 1.091085, 4.36434, [])
GPT_o1 = LLM("GPT-o1", "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", "", "OpenAI", 109.1085, 436.434, [])
GPT_o1_mini = LLM("GPT-o1-mini", "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", "", "OpenAI", 8.00129, 32.00516, [])
GPT_o3_mini = LLM("GPT-o3-mini", "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", "", "OpenAI", 8.00129, 32.00516, [])
GPT_o3_mini_high = LLM("GPT-o3-mini-high", "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", "", "OpenAI", 8.00129, 32.00516, [])

Claude_3_7_sonnet = LLM("Claude-3.7-sonnet", "", "", "Anthropic", 21.8217, 109.1085, [])

Gemini_2_0_flash_001 = LLM("Gemini-2.0-flash-001", "", "", "Google AI Studio", 0.72739, 2.90956, [])
Gemini_2_0_pro_exp_02_05 = LLM("Gemini-2.0-pro-exp-02-05", "", "", "Google AI Studio", 0.0, 0.0, [])
Gemini_2_0_flash_thinking_exp = LLM("Gemini-2.0-flash-thinking-exp", "", "", "Google AI Studio", 0.0, 0.0, [])

Grok_beta = LLM("Grok-beta", "", "", "xAI", 36.3695, 109.1085, [])

llm_list = [DeepSeek_R1, DeepSeek_V3, Deepseek_R1_distill_qwen_7b, DeepSeek_R1_distill_qwen_32b, GPT_4o, GPT_4o_mini, GPT_o1, GPT_o1_mini, GPT_o3_mini, GPT_o3_mini_high, Claude_3_7_sonnet, Gemini_2_0_flash_001, Gemini_2_0_pro_exp_02_05, Gemini_2_0_flash_thinking_exp, Grok_beta]

llm = Evaluate_LLM(llm_list, 0.0)
llm.call_llm(DeepSeek_R1, 1.0)
p = llm.calculate_probability()
print(p)

    