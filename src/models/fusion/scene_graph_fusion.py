from src.utils.data_loader import SceneGraph
from copy import deepcopy

class SceneGraphFusion:

    
    def __init__(self):
        pass
    
    def fuse_action(self, vsg: SceneGraph, action_semantics) -> SceneGraph:
        tsg = action_semantics['tsg']
        print(type(tsg))

        fused_sg = deepcopy(vsg)
        
            
        entity_mapping = {}  
        if (len(tsg['objects'])>=1):
            
            vsg_relations = {
            (rel['source'], rel['target'], rel['relation'])
            for rel in vsg.relations
        }
        
            for tsg_rel in tsg['relations']:
                rel_tuple = (tsg_rel['source'], tsg_rel['target'], tsg_rel['relation'])
                
                if rel_tuple not in vsg_relations:
                    fused_sg.relations.append(tsg_rel)
                      
            tsg_objects = tsg["objects"]
            tsg_attrs = tsg.get("attributes", {})

            for tsg_idx, attrs in tsg_attrs.items():

                category = tsg_objects[tsg_idx]  
                if not attrs:
                    continue  
                for vsg_idx, vsg_category in enumerate(fused_sg.objects):
                    if vsg_category == category:  

                        key = str(vsg_idx)
                        if key not in fused_sg.attributes:
                            fused_sg.attributes[key] = []
                        for attr in attrs:
                            if attr not in fused_sg.attributes[key]:
                                fused_sg.attributes[key].append(attr)
        
            
            for tsg_idx in range(min(len(tsg['objects']), len(vsg.objects))):
                vsg_idx = tsg_idx
                entity_mapping[tsg_idx] = vsg_idx
        
            print(f"  Entity mapping: {entity_mapping}")
        
            for tsg_idx, vsg_idx in entity_mapping.items():
                entity_name = tsg['objects'][tsg_idx] 
                
                key = str(vsg_idx) 

                if entity_name not in fused_sg.attributes[key]:
                    fused_sg.attributes[key].append(entity_name)
                    print(f"  [Fusion] Add attribute to entity {vsg_idx}: '{entity_name}'")
        else:
            tsg_to_fused_mapping = {}

            used_vsg_indices = set()
            
            print(f"\n  [Entity Matching & Adding]")
            
            for tsg_idx, tsg_obj in enumerate(tsg['objects']):
                matching_vsg_idx = self._find_matching_vsg_entity(tsg_obj, fused_sg.objects)

                if matching_vsg_idx is not None and matching_vsg_idx not in used_vsg_indices:
                    fused_idx = matching_vsg_idx
                    used_vsg_indices.add(matching_vsg_idx)

                    tsg_to_fused_mapping[tsg_idx] = fused_idx

                    fused_key = str(fused_idx)

                    if fused_key not in fused_sg.attributes:
                        fused_sg.attributes[fused_key] = []

                    if tsg_obj.lower() != fused_sg.objects[fused_idx].lower():
                        if tsg_obj not in fused_sg.attributes[fused_key]:
                            fused_sg.attributes[fused_key].append(tsg_obj)

                    if tsg_idx in tsg['attributes']:
                        tsg_attr = tsg['attributes'][tsg_idx]
                        
                        if 'modifiers' in tsg_attr:
                            for mod in tsg_attr['modifiers']:
                                if mod not in fused_sg.attributes[fused_key]:
                                    fused_sg.attributes[fused_key].append(mod)
                        
                        if 'role' in tsg_attr:
                            role = tsg_attr['role']
                            if role not in fused_sg.attributes[fused_key]:
                                fused_sg.attributes[fused_key].append(role)
                    
                    print(f"      Updated attributes: {fused_sg.attributes[fused_key]}")

                else:
                    fused_idx = len(fused_sg.objects)
                    
                    print(f"    Adding NEW entity [{fused_idx}]: '{tsg_obj}'")

                    fused_sg.objects.append(tsg_obj)

                    tsg_to_fused_mapping[tsg_idx] = fused_idx
            
            for tsg_rel in tsg['relations']:
                tsg_src = tsg_rel['source']
                tsg_tgt = tsg_rel['target']
                relation = tsg_rel['relation']

                fused_src = tsg_to_fused_mapping.get(tsg_src)
                fused_tgt = tsg_to_fused_mapping.get(tsg_tgt)
                
                if fused_src is not None and fused_tgt is not None:

                    rel_exists = any(
                        r['source'] == fused_src and 
                        r['target'] == fused_tgt and 
                        r['relation'] == relation
                        for r in fused_sg.relations
                    )
                    
                    if not rel_exists:
                        fused_sg.relations.append({
                            'source': fused_src,
                            'target': fused_tgt,
                            'relation': relation
                        })
                        
                        src_name = fused_sg.objects[fused_src]
                        tgt_name = fused_sg.objects[fused_tgt]
                        print(f"    Added relation: {src_name}[{fused_src}] --[{relation}]--> {tgt_name}[{fused_tgt}]")
                    else:
                        print(f"    Relation already exists: [{fused_src}] --[{relation}]--> [{fused_tgt}]")
            
            if hasattr(fused_sg, 'objects_id') and fused_sg.objects_id:
                num_new_entities = len(fused_sg.objects) - len(vsg.objects)
                if num_new_entities > 0:
                    fused_sg.objects_id.extend([0] * num_new_entities)
            
        return fused_sg
    
    def _find_matching_vsg_entity(self, tsg_obj, vsg_objects):
        tsg_obj_lower = tsg_obj.lower()
    
        for vsg_idx, vsg_obj in enumerate(vsg_objects):
            vsg_obj_lower = vsg_obj.lower()

            if tsg_obj_lower == vsg_obj_lower:
                return vsg_idx
        
        return None
