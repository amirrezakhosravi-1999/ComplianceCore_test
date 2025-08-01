#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for knowledge graph construction.
This includes relation extraction and graph building from regulatory text.
"""

import os
import json
import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from typing import List, Dict, Any, Tuple, Optional, Set
from pathlib import Path
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Class for building and managing knowledge graph of regulatory requirements."""
    
    def __init__(self,
                 fine_tuned_model_path: Optional[str] = None,
                 base_model_name: str = 'mistralai/Mistral-7B-Instruct-v0.2'):
        """
        Initialize the KnowledgeGraph class.
        
        Args:
            fine_tuned_model_path: Path to fine-tuned model (if available)
            base_model_name: Name of the base LLM model
        """
        self.fine_tuned_model_path = fine_tuned_model_path
        self.base_model_name = base_model_name
        self.model = None
        self.tokenizer = None
        self.graph = nx.DiGraph()
        
        logger.info(f"KnowledgeGraph initialized with fine_tuned_model_path={fine_tuned_model_path}, "
                   f"base_model_name={base_model_name}")
        
    def load_model(self):
        """Load LLM model for relation extraction."""
        logger.info("Loading model for relation extraction")
        
        if self.model is not None:
            logger.info("Model already loaded")
            return self.model, self.tokenizer
        
        try:
            if self.fine_tuned_model_path and os.path.exists(self.fine_tuned_model_path):
                logger.info(f"Loading fine-tuned model from {self.fine_tuned_model_path}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.fine_tuned_model_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.fine_tuned_model_path,
                    trust_remote_code=True
                )
            else:
                logger.info(f"Loading base model {self.base_model_name}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model_name,
                    trust_remote_code=True
                )
                
            # Ensure the tokenizer has required tokens
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Model loaded successfully")
            return self.model, self.tokenizer
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
    def extract_relations(self, text: str) -> List[Dict]:
        """
        Extract relations from text using the LLM.
        
        Args:
            text: Input text to extract relations from
            
        Returns:
            List of extracted relations
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Extract entities first using regex patterns for simplicity
        # In a production system, this would be replaced with a more sophisticated NER system
        entity_patterns = {
            'SYSTEM': r'([A-Za-z\s]+(?:system|pump|valve|reactor|container|vessel|piping))',
            'PARAMETER': r'([A-Za-z\s]+(?:limit|parameter|requirement|temperature|pressure|flow|value|condition)s?)',
            'EVENT': r'([A-Za-z\s]+(?:accident|event|incident|occurrence|transient|condition|basis)s?)',
            'REGULATION': r'([A-Za-z\s]+(?:regulation|requirement|standard|guideline|criterion|rule|code)s?)',
            'ACTION': r'([A-Za-z\s]+(?:action|operation|function|procedure|process|measure|control|test)s?)'
        }
        
        entities = []
        for entity_type, pattern in entity_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_text = match.group(1).strip()
                if len(entity_text) > 3 and entity_text not in [e['span'] for e in entities]:
                    entities.append({
                        'span': entity_text,
                        'type': entity_type
                    })
        
        # If using fine-tuned model, format input for relation extraction
        if self.fine_tuned_model_path and os.path.exists(self.fine_tuned_model_path):
            prompt = f"""Extract relations between entities from the following text.
Identify all entity pairs and their relationship types.

Text: {text}

Entities: """

            for entity in entities:
                prompt += f"{entity['span']} ({entity['type']}), "
            prompt = prompt.rstrip(', ')
            prompt += "\n\nRelations:\n"
            
        else:
            # For base model, use a more detailed prompt
            prompt = f"""As an expert in nuclear regulations, please analyze the relationships between entities in this text.
Extract specific relationships between each pair of entities, using precise relation types.

Text: "{text}"

Entities identified:
"""
            for i, entity in enumerate(entities):
                prompt += f"{i+1}. {entity['span']} (Type: {entity['type']})\n"
                
            prompt += """
For each relationship you identify, output in this format:
- [Entity 1] -> [RELATION_TYPE] -> [Entity 2]

Where RELATION_TYPE should be specific and descriptive (e.g., "requires", "constrains", "is_part_of", "measures", "prevents").

Identified Relations:
"""

        # Generate relations using the LLM
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False,
                num_return_sequences=1
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the generated part only
        if self.fine_tuned_model_path:
            # For fine-tuned model, extract only the generated relations
            response_part = response.split("Relations:\n")[-1].strip()
        else:
            # For base model, extract after the prompt
            response_part = response[len(prompt):].strip()
        
        # Parse relations from the LLM output
        relations = []
        relation_pattern = r'[\-•*]?\s*([^->\n]+)\s*->\s*([^->\n]+)\s*->\s*([^->\n]+)'
        matches = re.finditer(relation_pattern, response_part)
        
        for match in matches:
            head = match.group(1).strip()
            relation_type = match.group(2).strip()
            tail = match.group(3).strip()
            
            # Clean up any extra markings
            for item in [head, relation_type, tail]:
                item = re.sub(r'^\W+|\W+$', '', item)
            
            relations.append({
                'head': head,
                'type': relation_type,
                'tail': tail
            })
                
        logger.info(f"Extracted {len(relations)} relations from text")
        return relations
    
    def add_to_graph(self, 
                    semantic_unit: Dict[str, Any], 
                    relations: List[Dict]) -> None:
        """
        Add relations from a semantic unit to the graph.
        
        Args:
            semantic_unit: Semantic unit dictionary
            relations: List of relations extracted from the unit
        """
        unit_id = semantic_unit.get('unit_id', 'unknown')
        doc_id = semantic_unit.get('doc_id', 'unknown')
        section_title = semantic_unit.get('section_title', 'Unknown Section')
        page_number = semantic_unit.get('page_number', 0)
        text = semantic_unit.get('text', '')
        
        # Add relations to the graph
        for relation in relations:
            head = relation['head']
            rel_type = relation['type']
            tail = relation['tail']
            
            # Add nodes if they don't exist
            if not self.graph.has_node(head):
                self.graph.add_node(head, label=head, type="entity", mentions=1)
            else:
                self.graph.nodes[head]['mentions'] += 1
                
            if not self.graph.has_node(tail):
                self.graph.add_node(tail, label=tail, type="entity", mentions=1)
            else:
                self.graph.nodes[tail]['mentions'] += 1
            
            # Add edge with reference to the semantic unit
            if self.graph.has_edge(head, tail):
                # Update existing edge
                self.graph[head][tail]['weight'] += 1
                self.graph[head][tail]['references'].append({
                    'unit_id': unit_id,
                    'doc_id': doc_id,
                    'section': section_title,
                    'page': page_number,
                    'text': text
                })
            else:
                # Add new edge
                self.graph.add_edge(head, tail, 
                                   type=rel_type, 
                                   weight=1, 
                                   references=[{
                                       'unit_id': unit_id,
                                       'doc_id': doc_id,
                                       'section': section_title,
                                       'page': page_number,
                                       'text': text
                                   }])
    
    def build_graph_from_semantic_units(self, semantic_units: List[Dict[str, Any]]) -> nx.DiGraph:
        """
        Build a knowledge graph from semantic units.
        
        Args:
            semantic_units: List of semantic units
        
        Returns:
            NetworkX DiGraph with relations from semantic units
        """
        logger.info(f"Building knowledge graph from {len(semantic_units)} semantic units")
        
        # Create a new graph
        self.graph = nx.DiGraph()
        
        # Process each semantic unit
        for i, unit in enumerate(semantic_units):
            text = unit.get('text', '')
            if not text:
                continue
                
            logger.info(f"Processing unit {i+1}/{len(semantic_units)}: {text[:50]}...")
            
            # Extract relations from the unit
            relations = self.extract_relations(text)
            
            # Add relations to the graph
            self.add_to_graph(unit, relations)
            
            # Log progress every 10 units
            if (i + 1) % 10 == 0 or i == len(semantic_units) - 1:
                logger.info(f"Processed {i+1}/{len(semantic_units)} units, graph has {len(self.graph.nodes())} nodes and {len(self.graph.edges())} edges")
        
        logger.info(f"Knowledge graph built with {len(self.graph.nodes())} nodes and {len(self.graph.edges())} edges")
        return self.graph
    
    def save_graph(self, output_path: str) -> None:
        """
        Save the knowledge graph to a file.
        
        Args:
            output_path: Path to save the graph
        """
        logger.info(f"Saving knowledge graph to {output_path}")
        
        # Convert the graph to a dictionary
        data = {
            'nodes': [],
            'edges': []
        }
        
        # Add nodes with attributes
        for node, attrs in self.graph.nodes(data=True):
            node_data = {'id': node}
            node_data.update(attrs)
            data['nodes'].append(node_data)
            
        # Add edges with attributes
        for u, v, attrs in self.graph.edges(data=True):
            edge_data = {'source': u, 'target': v}
            
            # Process references to ensure they are serializable
            if 'references' in attrs:
                refs = []
                for ref in attrs['references']:
                    refs.append({k: str(v) if isinstance(v, Path) else v for k, v in ref.items()})
                attrs['references'] = refs
                
            edge_data.update(attrs)
            data['edges'].append(edge_data)
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Knowledge graph saved to {output_path}")
        
    def load_graph(self, input_path: str) -> nx.DiGraph:
        """
        Load the knowledge graph from a file.
        
        Args:
            input_path: Path to load the graph from
            
        Returns:
            Loaded NetworkX DiGraph
        """
        logger.info(f"Loading knowledge graph from {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Create a new graph
        self.graph = nx.DiGraph()
        
        # Add nodes with attributes
        for node_data in data['nodes']:
            node_id = node_data.pop('id')
            self.graph.add_node(node_id, **node_data)
            
        # Add edges with attributes
        for edge_data in data['edges']:
            source = edge_data.pop('source')
            target = edge_data.pop('target')
            self.graph.add_edge(source, target, **edge_data)
            
        logger.info(f"Knowledge graph loaded with {len(self.graph.nodes())} nodes and {len(self.graph.edges())} edges")
        return self.graph
    
    def get_graph_layout(self, graph=None) -> Dict[str, Tuple[float, float]]:
        """
        Get a layout for graph visualization.
        
        Args:
            graph: NetworkX graph to get layout for (defaults to self.graph)
            
        Returns:
            Dictionary mapping node IDs to positions
        """
        if graph is None:
            graph = self.graph
            
        # Use spring layout for better visualization
        pos = nx.spring_layout(graph, k=0.15, iterations=20)
        return pos
    
    def draw_graph_matplotlib(self, graph=None, pos=None, ax=None) -> None:
        """
        Draw the graph using matplotlib.
        
        Args:
            graph: NetworkX graph to draw (defaults to self.graph)
            pos: Node positions for drawing
            ax: Matplotlib axis to draw on
        """
        if graph is None:
            graph = self.graph
            
        if pos is None:
            pos = self.get_graph_layout(graph)
            
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 10))
            
        # Draw nodes
        nx.draw_networkx_nodes(
            graph, pos, 
            node_size=700, 
            node_color='lightblue',
            alpha=0.8,
            ax=ax
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            graph, pos,
            width=[data['weight'] * 0.5 for _, _, data in graph.edges(data=True)],
            alpha=0.5,
            edge_color='gray',
            arrows=True,
            arrowsize=10,
            connectionstyle='arc3,rad=0.1',
            ax=ax
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            graph, pos,
            font_size=8,
            font_family='sans-serif',
            ax=ax
        )
        
        # Draw edge labels (relation types)
        edge_labels = {(u, v): d['type'] for u, v, d in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(
            graph, pos,
            edge_labels=edge_labels,
            font_size=6,
            font_color='red',
            ax=ax
        )
        
        # Set plot properties
        ax.set_title("Nuclear Regulatory Knowledge Graph")
        ax.axis('off')
        
    def find_dependencies(self, entity: str, max_depth: int = 3) -> Dict[str, Any]:
        """
        Find dependencies for a given entity in the graph.
        
        Args:
            entity: Entity to find dependencies for
            max_depth: Maximum depth to search for dependencies
            
        Returns:
            Dictionary with dependencies information
        """
        if not self.graph.has_node(entity):
            logger.warning(f"Entity '{entity}' not found in the graph")
            return {'entity': entity, 'found': False}
            
        # Get all predecessors (dependencies) of the entity up to max_depth
        deps = {}
        visited = {entity}
        current_level = {entity}
        
        for depth in range(1, max_depth + 1):
            next_level = set()
            for node in current_level:
                for pred in self.graph.predecessors(node):
                    if pred not in visited:
                        edge_data = self.graph.get_edge_data(pred, node)
                        
                        # Create dependency entry
                        if pred not in deps:
                            deps[pred] = {
                                'relation': edge_data.get('type', 'related_to'),
                                'weight': edge_data.get('weight', 1),
                                'references': edge_data.get('references', []),
                                'depth': depth
                            }
                        visited.add(pred)
                        next_level.add(pred)
            
            current_level = next_level
            if not current_level:
                break
                
        return {
            'entity': entity,
            'found': True,
            'dependencies': deps,
            'total_dependencies': len(deps)
        }
    
    def find_related_regulations(self, entity: str, relation_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Find regulatory clauses related to a given entity.
        
        Args:
            entity: Entity to find related regulations for
            relation_types: Filter by specific relation types
            
        Returns:
            List of related regulations with metadata
        """
        if not self.graph.has_node(entity):
            logger.warning(f"Entity '{entity}' not found in the graph")
            return []
            
        relations = []
        
        # Check all edges connected to the entity
        for src, tgt, data in self.graph.edges(data=True):
            # Check if this edge involves our entity
            if src == entity or tgt == entity:
                # Check relation type if filtering is requested
                rel_type = data.get('type', '')
                if relation_types and rel_type not in relation_types:
                    continue
                    
                # Get the other entity (not the one we're looking for)
                other = tgt if src == entity else src
                
                # Direction of the relation
                direction = 'outgoing' if src == entity else 'incoming'
                
                # Add to relations list
                for ref in data.get('references', []):
                    relations.append({
                        'entity': entity,
                        'related_entity': other,
                        'relation_type': rel_type,
                        'direction': direction,
                        'weight': data.get('weight', 1),
                        'unit_id': ref.get('unit_id', ''),
                        'doc_id': ref.get('doc_id', ''),
                        'section': ref.get('section', ''),
                        'page': ref.get('page', 0),
                        'text': ref.get('text', '')
                    })
                    
        return relations


def main():
    """Main function to demonstrate the knowledge graph functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build a knowledge graph from semantic units')
    parser.add_argument('--input', type=str, default='../data/semantic_units.json',
                        help='Path to input semantic units JSON file')
    parser.add_argument('--output', type=str, default='../output/knowledge_graph.json',
                        help='Path to output knowledge graph JSON file')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to fine-tuned model')
    parser.add_argument('--base-model', type=str, default='mistralai/Mistral-7B-Instruct-v0.2',
                        help='Name of base model')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit the number of semantic units to process (0 = no limit)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate a visualization of the graph')
    args = parser.parse_args()
    
    # Initialize knowledge graph
    kg = KnowledgeGraph(
        fine_tuned_model_path=args.model,
        base_model_name=args.base_model
    )
    
    # Load semantic units
    with open(args.input, 'r', encoding='utf-8') as f:
        semantic_units = json.load(f)
        
    # Limit units if requested
    if args.limit > 0:
        semantic_units = semantic_units[:args.limit]
        logger.info(f"Limited to first {args.limit} semantic units")
        
    # Build graph
    graph = kg.build_graph_from_semantic_units(semantic_units)
    
    # Save graph
    kg.save_graph(args.output)
    
    # Generate visualization if requested
    if args.visualize:
        output_img = args.output.replace('.json', '.png')
        logger.info(f"Generating visualization to {output_img}")
        
        plt.figure(figsize=(20, 18))
        pos = kg.get_graph_layout()
        kg.draw_graph_matplotlib(pos=pos)
        plt.tight_layout()
        plt.savefig(output_img, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Visualization saved to {output_img}")
    
    logger.info("Knowledge graph processing completed successfully!")


if __name__ == "__main__":
    main() 