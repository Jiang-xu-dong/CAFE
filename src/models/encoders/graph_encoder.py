
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from src.utils.data_loader import SceneGraph

class SceneGraphEncoder(nn.Module):
    def __init__(self, hidden_dim=256, output_dim=512, dropout=0.3, device='cuda'):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.object_vocab = self._build_object_vocab()
        self.attribute_vocab = self._build_attribute_vocab()
        self.relation_vocab = self._build_relation_vocab()

        self.object_embedding = nn.Embedding(len(self.object_vocab), 256, padding_idx=0)  
        self.attribute_embedding = nn.Embedding(len(self.attribute_vocab), 256, padding_idx=0) 
        self.id_embedding = nn.Embedding(
            num_embeddings=10,  
            embedding_dim=256,  
            padding_idx=0
        )
        self.node_fusion = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim*2)
        self.gcn3 = GCNConv(hidden_dim*2, output_dim)

        self.readout = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim*2)
        self.norm3 = nn.LayerNorm(output_dim)
        self.to(device)
    
    
    def forward(self, scene_graph: SceneGraph) -> torch.Tensor:

        node_features = self._build_node_features(scene_graph)
        

        edge_index = self._build_edge_index(scene_graph)
        

        x = self.gcn1(node_features, edge_index) 
        x = self.norm1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.gcn2(x, edge_index)  
        x = self.norm2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.gcn3(x, edge_index)  # (N, 512)
        x = self.norm3(x)

        x = self.dropout(x)
        graph_emb = torch.mean(x, dim=0, keepdim=True)  # (1, 512)
        

        output = self.readout(graph_emb)  # (1, 512)
        
        return output
    
    
    
    def _encode_object_ids(self, scene_graph, num_nodes):

        if hasattr(scene_graph, 'objects_id') and scene_graph.objects_id:
            objects_id = scene_graph.objects_id
            
            if len(objects_id) < num_nodes:
                objects_id = objects_id + [0] * (num_nodes - len(objects_id))
            elif len(objects_id) > num_nodes:
                objects_id = objects_id[:num_nodes]
            
            id_tensor = torch.tensor(objects_id, dtype=torch.long).to(self.device) 

            id_emb = self.id_embedding(id_tensor)  # (num_nodes, hidden_dim)
            

            
            
            return id_emb
        else:
            return torch.zeros(num_nodes, self.hidden_dim).to(self.device)
    
    
     
    def _encode_attributes(self, scene_graph, num_nodes):

        if not scene_graph.attributes:
            return torch.zeros(num_nodes, self.hidden_dim).to(self.device)
        
        attr_embeddings = []
        
        for i in range(num_nodes):
            node_attrs = scene_graph.attributes.get(str(i), [])
            
            if not node_attrs:
                attr_embeddings.append(
                    torch.zeros(self.hidden_dim).to(self.device)
                )
            else:
                attr_indices = []
                for attr in node_attrs:
                    attr_idx = self.attribute_vocab.get(attr.lower(), 1)  
                    attr_indices.append(attr_idx)

                attr_tensor = torch.tensor(attr_indices, dtype=torch.long).to(self.device)
                 
                attr_embs = self.attribute_embedding(attr_tensor) 
                
                attr_emb = torch.mean(attr_embs, dim=0)  
                
                attr_embeddings.append(attr_emb)
        attr_embeddings = torch.stack(attr_embeddings)  # (num_nodes, hidden_dim)
        
        return attr_embeddings
    def _build_object_vocab(self):

        objects = YOUR_OBJECT_LIST 
        return {obj: idx for idx, obj in enumerate(objects)}
    
    def _build_attribute_vocab(self):
        attributes = YOUR_ATTRIBUTE_LIST
    
    def _build_relation_vocab(self):
        relations = YOUR_RELATION_LIST
        return {rel: idx for idx, rel in enumerate(relations)}
    
    
    def _build_node_features(self, scene_graph: SceneGraph) -> torch.Tensor:
        num_nodes = len(scene_graph.objects)
        
        object_indices = [
            self.object_vocab.get(obj.lower(), 1) for obj in scene_graph.objects
        ]  
        
        object_emb = self.object_embedding(
            torch.tensor(object_indices, device=self.device)
        )
        id_emb = self._encode_object_ids(scene_graph, num_nodes)
        attr_emb = self._encode_attributes(scene_graph, num_nodes)

        node_features = torch.cat([object_emb, attr_emb,id_emb], dim=-1)
        print("node_features.shape =", node_features.shape)
        if num_nodes > 1:
            node_features = self.node_fusion(node_features)
        else:
            node_features = self.node_fusion(node_features)

        
        return node_features
    
    def _build_edge_index(self, scene_graph: SceneGraph) -> torch.Tensor:
        edge_list = []
        
        for rel in scene_graph.relations:
            source = rel['source']
            target = rel['target']
            
            if source < len(scene_graph.objects) and target < len(scene_graph.objects):
                edge_list.append([source, target])
                edge_list.append([target, source]) 
        
        if not edge_list:
            num_nodes = len(scene_graph.objects)
            edge_list = [[i, i] for i in range(num_nodes)]
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(self.device)
        return edge_index
    