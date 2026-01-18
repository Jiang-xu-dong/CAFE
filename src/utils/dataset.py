import torch
from torch.utils.data import Dataset
from typing import List, Dict
from src.utils.data_loader import SceneGraphLoader, SceneGraph

class NSFWDataset(Dataset):

    def __init__(self, json_path: str):
        self.loader = SceneGraphLoader(json_path)
        self.video_keys = self.loader.get_video_keys()
        self.class_to_idx = {
            'normal': 0,
            'illegal_activity': 1,
            'harassment': 2,
            'hateful': 3,
            'self-harm':4,
            'sexual':5,
            'conflict':6,
            'threaten':7
        }
        self.num_classes = len(self.class_to_idx)
    
    def __len__(self):
        return len(self.video_keys)
    
    def __getitem__(self, idx):
        video_key = self.video_keys[idx]
        scene_graph = self.loader.get_scene_graph(video_key)
        label_str = scene_graph.nsfw_labels[0] 
        label_idx = self.class_to_idx.get(label_str, 0)  
        
        return scene_graph, scene_graph.text, label_idx
    
    def _compute_label(self, nsfw_labels: List[str]) -> float:
        if not nsfw_labels or nsfw_labels == ['safe']:
            return 0.1
        scores = [self.label_mapping.get(label, 0.5) for label in nsfw_labels]
        return max(scores)
    
    def collate_fn(self, batch):
        scene_graphs, texts, labels = zip(*batch)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return list(scene_graphs), list(texts), labels