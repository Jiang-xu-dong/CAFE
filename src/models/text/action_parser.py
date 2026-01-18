from dataclasses import dataclass
from src.utils.data_loader import SceneGraph
import spacy
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

@dataclass
class ActionSemantics:
    tsg: SceneGraph
    intent: str
    confidence: float
    entities_attributes: dict


class ActionParser:
    def __init__(self, device='cuda'):
        self.device = device
        try:
            self.nlp = spacy.load('en_core_web_trf')
        except:
            print("Loading en_core_web_trf...")
            self.nlp = spacy.load('en_core_web_trf')


        self.all_intent_labels = []
        self.label_to_intent = {} 
        for intent, labels in self.intent_labels.items():
            for label in labels:
                self.all_intent_labels.append(label)
                self.label_to_intent[label] = intent
    
        
    def _extract_adjectives(self, doc, semantic_roles):
        adjectives = {}
        
        for role_idx, role in enumerate(semantic_roles):
            agent = role['agent']
            patient = role['patient']
            instrument = role['instrument']
            if agent:
                agent_attrs = self._get_modifiers(agent)
                if agent_attrs:
                    adjectives[role_idx * 3] = agent_attrs 
            if patient:
                patient_attrs = self._get_modifiers(patient)
                if patient_attrs:
                    adjectives[role_idx * 3 + 1] = patient_attrs 

            if instrument:
                instrument_attrs = self._get_modifiers(instrument)
                if instrument_attrs:
                    adjectives[role_idx * 3 + 2] = instrument_attrs  
        
        return adjectives
    
    def _get_modifiers(self, token):
        modifiers = []

        for child in token.children:
            if child.dep_ == 'amod':  
                modifiers.append(child.text.lower())
        for child in token.children:
            if child.dep_ == 'compound':  
                modifiers.append(child.text.lower())

        if token.i > 0:
            left_token = token.doc[token.i - 1]
            if left_token.pos_ == 'ADJ' and left_token.text.lower() not in modifiers:
                modifiers.append(left_token.text.lower())
        return modifiers
    
    def _extract_semantic_roles(self, doc):
        roles = []
        
        for token in doc:

            
            if token.pos_ == 'VERB':
                role = {
                    'action': token,
                    'agent': None,
                    'patient': None,
                    'instrument': None,
                    'action_phrase': '',
                    'full_clause': ''  
                }

                for child in token.children:
                    if child.dep_ in ['nsubj', 'nsubjpass']:
                        role['agent'] = child
                    elif child.dep_ in ['dobj', 'pobj']:
                        role['patient'] = child

                    elif child.dep_ == 'prep':
                        for grandchild in child.children:
                            if grandchild.dep_ == 'pobj':
                                role['instrument'] = grandchild
                action_tokens = [token]
                for child in token.children:
                    if child.dep_ in ['advmod', 'prep']:
                        action_tokens.extend([child] + list(child.subtree))

                if role['patient']:
                    action_tokens.extend(list(role['patient'].subtree))

                action_tokens = sorted(set(action_tokens), key=lambda t: t.i)
                role['action_phrase'] = ' '.join([t.text for t in action_tokens])

                clause_tokens = []
                if role['agent']:
                    clause_tokens.extend(list(role['agent'].subtree))
                clause_tokens.extend(action_tokens)
                clause_tokens = sorted(set(clause_tokens), key=lambda t: t.i)
                role['full_clause'] = ' '.join([t.text for t in clause_tokens])
                
                roles.append(role)
        
        return roles


    
    def _get_entity_span_text(self, token):
        span_tokens = list(token.subtree)
        span_tokens.sort(key=lambda t: t.i)
        return ' '.join([t.text for t in span_tokens])
    
    def _get_entity_text(self, token):
        if token is None:
            return None
        return token.text
    
    def __call__(self, text: str):
        print(f"\n{'='*70}")
        print(f"Parsing: {text}")
        print(f"{'='*70}")
        
        doc = self.nlp(text)
        semantic_roles = self._extract_semantic_roles(doc)
        
        if not semantic_roles:

            return {
                'tsg': {
                    'objects': [],
                    'relations': [],
                    'attributes': {}
                },
                'intent': 'unknown',
                'confidence': 0.0
            }
        
        for i, role in enumerate(semantic_roles):
            print(f"    Role {i}: agent={role['agent']}, action={role['action']}, "
                  f"patient={role['patient']}, instrument={role['instrument']}")

        semantic_roles = self._merge_semantic_roles(semantic_roles)
        
        print("\n  [Merged Semantic Roles]")
        for i, role in enumerate(semantic_roles):
            print(f"    Role {i}: agent={role['agent']}, action={role['action']}, "
                  f"patient={role['patient']}, instrument={role['instrument']}")
        tsg = self._build_tsg_with_modifiers(doc, semantic_roles)
    
        
        print(f"\n{'='*70}")
        print(f"Result:")
        print(f"  TSG: {tsg}")
        print(f"{'='*70}\n")
        
        return {
            'tsg': tsg,
        }
    
    def _build_tsg_with_modifiers(self, doc, semantic_roles):
        objects = []
        relations = []
        attributes = {}
        entity_to_idx = {}
        
        def add_entity(token):
            if token is None:
                return None
            
            entity_text = self._get_entity_text(token)
            
            if entity_text not in entity_to_idx:
                idx = len(objects)
                objects.append(entity_text.lower())
                attributes[idx] = []
                modifiers = self._get_modifiers(token)
                if modifiers:
                    attributes[idx].extend(modifiers)
                    print(f"    Added entity [{idx}]: '{entity_text}' with modifiers {modifiers}")
                else:
                    print(f"    Added entity [{idx}]: '{entity_text}'")
                
                entity_to_idx[entity_text] = idx
            
            return entity_to_idx[entity_text]
        for role_idx, role in enumerate(semantic_roles):
            agent = role['agent']
            action = role['action']
            patient = role['patient']
            instrument = role['instrument']
            agent_idx = add_entity(agent)
            patient_idx = add_entity(patient)
            instrument_idx = add_entity(instrument)
            action_name = action.lemma_.lower() if action else 'interact'

            if agent_idx is not None and patient_idx is not None:
                relations.append({
                    'source': agent_idx,
                    'target': patient_idx,
                    'relation': action_name
                })
                
                src_name = objects[agent_idx]
                tgt_name = objects[patient_idx]
                print(f"    Added relation: {src_name}[{agent_idx}] --[{action_name}]--> {tgt_name}[{patient_idx}]")
            if agent_idx is not None and instrument_idx is not None:
                relations.append({
                    'source': agent_idx,
                    'target': instrument_idx,
                    'relation': 'use'
                })
                
                src_name = objects[agent_idx]
                inst_name = objects[instrument_idx]
                print(f"    Added relation: {src_name}[{agent_idx}] --[use]--> {inst_name}[{instrument_idx}]")
        
        tsg = {
            "objects": objects,
            "relations": relations,
            "attributes": attributes
        }
        
        return tsg
    
    def _merge_semantic_roles(self, roles):
        merged_roles = []
        
        i = 0
        while i < len(roles):
            role = roles[i]
            if i + 1 < len(roles):
                next_role = roles[i + 1]

                if next_role['action'] and next_role['action'].lemma_.lower():
                    if next_role['patient'] and not role['instrument']:
                        role['instrument'] = next_role['patient']
                    i += 2
                    merged_roles.append(role)
                    continue
            
            merged_roles.append(role)
            i += 1
        
        return merged_roles
    
    def _get_entity_text(self, token):
        if token is None:
            return None
        return token.text
    
    def _get_modifiers(self, token):
        if token is None:
            return []
        
        modifiers = []
        for child in token.children:
            if child.dep_ == 'amod':
                modifiers.append(child.text.lower())

        for child in token.children:
            if child.dep_ == 'compound':
                modifiers.append(child.text.lower())
        if token.i > 0:
            left_token = token.doc[token.i - 1]
            if left_token.pos_ == 'ADJ' and left_token.text.lower() not in modifiers:
                modifiers.append(left_token.text.lower())
        
        return modifiers