import torch.nn as nn
from transformers import GPT2PreTrainedModel, GPT2LMHeadModel

class GPT2WithRegression(GPT2PreTrainedModel):
    def __init__(self, config):
        super(GPT2WithRegression, self).__init__(config)
        self.gpt2 = GPT2LMHeadModel(config)
        self.dropout = nn.Dropout(config.attn_pdrop)
        self.regression = nn.Linear(config.n_embd, 1)

    def forward(self, **kwargs):
        kwargs["output_hidden_states"] = True
        gpt2_outputs = self.gpt2(**kwargs)
        hidden_states = gpt2_outputs.hidden_states[-1]
        output = self.dropout(hidden_states)
        output = self.regression(output).squeeze(-1)
        return gpt2_outputs.logits, output