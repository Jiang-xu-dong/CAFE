import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import os

class BERTTextTypeClassifier(nn.Module):

    def __init__(self, model_name='google-bert/bert-base-uncased', num_classes=2, device='cuda'):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.num_classes = num_classes
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        self.type_names = ['action', 'context']
        self.to(device)
    
    def forward(self, text: str, return_logits: bool = False):
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.bert(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :] 

        logits = self.classifier(cls_embedding)  
        
        if return_logits:
            return logits, cls_embedding
        else:
            pred_idx = torch.argmax(logits, dim=-1).item()
            return self.type_names[pred_idx]
    
    def __call__(self, text: str) -> str:
        self.eval()
        with torch.no_grad():
            return self.forward(text, return_logits=False)
    
    def save_model(self, save_path: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'type_names': self.type_names
        }, save_path)
    
    