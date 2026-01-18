import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from tqdm import tqdm
import numpy as np
from datetime import datetime

from src.utils.dataset import NSFWDataset
from src.core.fusion_detector import FusionNSFWDetector
import json
class Trainer:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(
            self.config['device'] if torch.cuda.is_available() else 'cpu'
        )
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)
        self.model = FusionNSFWDetector(
            hidden_dim=self.config['model']['hidden_dim'],
            output_dim=self.config['model']['output_dim'],
            dropout=self.config['model']['dropout'],
            device=self.device
        ).to(self.device)

        self.train_dataset = NSFWDataset(self.config['data']['train_path'])
        self.val_dataset = NSFWDataset(self.config['data']['val_path'])
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            collate_fn=self.train_dataset.collate_fn,
            num_workers=self.config['training']['num_workers']
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            collate_fn=self.val_dataset.collate_fn,
            num_workers=self.config['training']['num_workers']
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        self.criterion = nn.CrossEntropyLoss()

        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0

        all_preds = []
        all_labels = []
        for scene_graphs, texts, labels in self.train_loader: 
            self.optimizer.zero_grad()
            batch_logits = []
            for sg, text in zip(scene_graphs, texts):
                result = self.model(sg, text)
                logit = result['logits']  
                batch_logits.append(logit)
            logits = torch.stack(batch_logits).to(self.device)  # [B, num_classes]
            labels = labels.to(self.device).long()              # [B]

            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        avg_loss = epoch_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        all_preds_np = np.array(all_preds)
        all_labels_np = np.array(all_labels)
        acc = np.mean(all_preds_np == all_labels_np)

        return avg_loss, acc

    
    
    def save_checkpoint(self, epoch, val_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        latest_path = os.path.join(
            self.config['checkpoint_dir'], 
            'latest.pth'
        )
        torch.save(checkpoint, latest_path)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(
                self.config['checkpoint_dir'], 
                'best.pth'
            )
            torch.save(checkpoint, best_path)
    
    def train(self):
        num_epochs = self.config['training']['num_epochs']
        for epoch in range(num_epochs):
            train_loss,acc,acc_action,acc_context = self.train_epoch(epoch)
            val_loss, acc = self.validate(epoch)
            self.scheduler.step(train_loss)
            self.save_checkpoint(epoch, train_loss)
        

def main():
    trainer = Trainer('config.yaml')
    trainer.train()


if __name__ == '__main__':
    main()