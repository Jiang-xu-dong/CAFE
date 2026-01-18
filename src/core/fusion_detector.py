import torch
import torch.nn as nn
from src.utils.data_loader import SceneGraph
from src.models.encoders.graph_encoder import SceneGraphEncoder
from src.models.text.text_classifier import BERTTextTypeClassifier
from src.models.text.action_parser import ActionParser
from src.models.text.background_parser import BackgroundParser
from src.models.fusion.scene_graph_fusion import SceneGraphFusion
import json

def tsg_to_scenegraph(tsg_dict, video_id="", text=""):
    tsg = tsg_dict.get("tsg", {})

    objects = tsg.get("objects", [])
    relations = tsg.get("relations", [])

    raw_attrs = tsg.get("attributes", {})
    attributes = {str(k): v for k, v in raw_attrs.items()}

    return SceneGraph(
        video_id=video_id,
        nsfw_labels=[],
        objects=objects,
        attributes=attributes,
        objects_id=list(range(len(objects))),
        relations=relations,
        text=text,
        background=None
    )



count=0
class FusionNSFWDetector(nn.Module):
    def __init__(self, hidden_dim=256, output_dim=512, dropout=0.3, device='cuda'):
        super().__init__()
        self.device = device
 
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
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)

        self.graph_encoder = SceneGraphEncoder(hidden_dim, output_dim, dropout, device)
        
        self.action_parser = ActionParser()
        self.background_parser = BackgroundParser()
        
        self.fusion_module = SceneGraphFusion()
        self.layer_norm = nn.LayerNorm(512)

        self.classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),  
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, self.num_classes)
        )

        
        
        self.fusion = nn.Linear(1280, 512)
        self.to(device)
    
    
    def forward(self, scene_graph: SceneGraph, text: str) -> dict:
        tsg = self.action_parser(text)
        print(tsg)
        global count
        count+=1

        fused_sg = self.fusion_module.fuse_action(scene_graph, tsg)

        print(fused_sg)

        sg_emb = self.graph_encoder(scene_graph)  
        fused = self.fusion(sg_emb,tsg)
        logits = self.classifier(fused).squeeze()  
        probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs).item()
        
        return {
            'logits': logits,
            'probs': probs,
            'predicted_class': predicted_class,
            'predicted_label': self.idx_to_class[predicted_class],
        }