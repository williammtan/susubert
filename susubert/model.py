import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

class BERTMatcher(nn.Module):
    def __init__(self, lm, device='cpu'): # task = train | eval
        super().__init__()
        self.device = device
        self.lm = lm

        self.bert = AutoModel.from_pretrained(lm).to(device)
        self.dropout = nn.Dropout(0.1)
        self.fully_connected = nn.Linear(768, 2) # bert final layer -> binary classification
    
    def forward(self, x):
        x = torch.squeeze(x, 1).to(self.device)

        if self.training:
            self.bert.train()
            output = self.bert(input_ids=x)
            pooled_output = output[0][:, 0, :]
            pooled_output = self.dropout(pooled_output)
        else:
            self.bert.val()
            output = self.bert(x)
            pooled_output = output[0][:, 0, :]

        logits = self.fully_connected(pooled_output)
        y_pred = logits.argmax(-1)

        return logits, y_pred
    
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.lm)


