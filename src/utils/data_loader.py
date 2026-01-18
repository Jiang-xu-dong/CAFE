import json
from dataclasses import dataclass, field
from typing import List, Dict, Any,Optional

@dataclass
class SceneGraph:
    video_id: str
    nsfw_labels: List[str]
    objects: List[str]
    attributes: Dict[str, List[str]]
    objects_id: List[int]
    relations: List[Dict[str, Any]]
    text: str = ""
    background: Optional[Dict] = None
    
    @classmethod
    def from_dict(cls, video_key: str, data: dict):
        scenes = data['scenes']
        return cls(
            video_id=data['video_id'],
            nsfw_labels=data.get('nsfw_labels', []),
            objects=scenes['objects'],
            attributes=scenes.get('attributes', []),
            objects_id=scenes.get('objects_id', []),
            relations=scenes.get('relations', []),
            text=data.get('text', ''),
            background=data.get('background', None)
        )
    
    def __repr__(self):
        return (f"SceneGraph(\n"
                f"  objects={self.objects},\n"
                f"  relations={self.relations},\n"
                f"  objects_id={self.objects_id},\n"
                f"  attributes={self.attributes},\n"
                f"  background={self.background}\n"
                f")")
    def to_dict(self):
        return {
            'video_id': self.video_id,
            'nsfw_labels': self.nsfw_labels,
            'objects': self.objects,
            'attributes': self.attributes,
            'objects_id': self.objects_id,
            'relations': self.relations,
            'text': self.text
        }
        
    @classmethod
    def empty(cls, video_id="null", text=""):
        return cls(
            video_id=video_id,
            nsfw_labels=[],
            objects=[],
            attributes={},
            objects_id=[],
            relations=[],
            text=text,
            background=None
        )


class SceneGraphLoader:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.data = self._load_json()
    
    def _load_json(self) -> Dict:

        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[Loader] Loaded {len(data)} videos from {self.json_path}")
        return data
    
    def get_scene_graph(self, video_key: str) -> SceneGraph:
        if video_key not in self.data:
            raise KeyError(f"Video key '{video_key}' not found")
        return SceneGraph.from_dict(video_key, self.data[video_key])
    
    def get_all_scene_graphs(self) -> List[SceneGraph]:
        return [SceneGraph.from_dict(key, data) 
                for key, data in self.data.items()]
    
    def get_video_keys(self) -> List[str]:
        return list(self.data.keys())