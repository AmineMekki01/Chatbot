from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

class HealthCareChatbotModel(nn.Module):
    def __init__(self, n_classes):
        super(HealthCareChatbotModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)
    
    # improve the model 
    def unfreeze(self):
        for param in self.bert.parameters():
            param.requires_grad = True
    
    def freeze(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    
