#!/usr/bin/env python3
"""
Load Specific Knowledge Graph Format
===================================
This script loads the specific knowledge_graph.pkl format with entity_map and relationship_map
"""

from graphrag import GraphRAG, Entity, Relation
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random

def load_specific_knowledge_graph():
    """Load the specific knowledge graph format and create 3D visualization"""
    print("Loading knowledge_graph.pkl (Specific Format)")
    print("=" * 50)
    
    # Load the pickle file
    try:
        with open("knowledge_graph.pkl", 'rb') as f:
            data = pickle.load(f)
        print("✓ Successfully loaded pickle file")
    except Exception as e:
        print(f"✗ Error loading pickle file: {e}")
        return None
    
    # Extract the components
    entity_map = data.get('entity_map', {})
    relationship_map = data.get('relationship_map', {})
    community_summaries = data.get('community_summaries', {})
    
    print(f"Found {len(entity_map)} entities")
    print(f"Found {len(relationship_map)} relationship entries")
    print(f"Found {len(community_summaries)} community summaries")
    
    # Initialize GraphRAG
    graph_rag = GraphRAG()
    
    # Convert entities
    print("\nConverting entities...")
    entity_count = 0
    for entity_name, entity_info in entity_map.items():
        if isinstance(entity_info, dict):
            entity = Entity(
                id=f"entity_{entity_count}",
                name=entity_name,
                type=entity_info.get('type', 'ENTITY'),
                properties=entity_info
            )
            graph_rag.add_entity(entity)
            entity_count += 1
            
            if entity_count % 1000 == 0:
                print(f"  Processed {entity_count} entities...")
    
    print(f"✓ Converted {entity_count} entities")
    
    # Convert relationships
    print("\nConverting relationships...")
    relation_count = 0
    entity_name_to_id = {entity.name: entity.id for entity in graph_rag.entities.values()}
    
    for source_entity, relationships in relationship_map.items():
        if source_entity in entity_name_to_id:
            source_id = entity_name_to_id[source_entity]
            
            if isinstance(relationships, list):
                for rel_info in relationships:
                    if isinstance(rel_info, dict):
                        target_entity = rel_info.get('target', '')
                        relation_type = rel_info.get('type', 'RELATED')
                        
                        if target_entity in entity_name_to_id:
                            target_id = entity_name_to_id[target_entity]
                            
                            relation = Relation(
                                id=f"rel_{relation_count}",
                                source=source_id,
                                target=target_id,
                                relation_type=relation_type,
                                properties=rel_info
                            )
                            graph_rag.add_relation(relation)
                            relation_count += 1
                            
                            if relation_count % 1000 == 0:
                                print(f"  Processed {relation_count} relationships...")
    
    print(f"✓ Converted {relation_count} relationships")
    
    # Show statistics
    stats = graph_rag.get_statistics()
    print(f"\nFinal Knowledge Graph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Sample the graph for visualization (it's too large)
    print(f"\nSampling graph for visualization...")
    sampled_graph = sample_large_graph(graph_rag, max_nodes=100)
    sampled_stats = sampled_graph.get_statistics()
    print(f"Sampled Graph Statistics:")
    for key, value in sampled_stats.items():
        print(f"  {key}: {value}")
    
    # Create 3D visualization
    print(f"\nGenerating 3D visualization...")
    try:
        sampled_graph.visualize_3d("knowledge_graph_3d.png")
        print("✓ 3D visualization completed!")
        
        # Also create 2D for comparison
        print("Generating 2D visualization...")
        sampled_graph.visualize_2d("knowledge_graph_2d.png")
        print("✓ 2D visualization completed!")
        
    except Exception as e:
        print(f"✗ Error during visualization: {e}")
    
    return graph_rag, sampled_graph

def sample_large_graph(graph_rag, max_nodes=100):
    """Sample a large graph to make it manageable for visualization"""
    print(f"Sampling {max_nodes} nodes from {len(graph_rag.entities)} entities...")
    
    # Create a new GraphRAG with sampled nodes
    sampled_graph = GraphRAG()
    
    # Sample entities randomly
    all_entities = list(graph_rag.entities.values())
    if len(all_entities) > max_nodes:
        sampled_entities = random.sample(all_entities, max_nodes)
    else:
        sampled_entities = all_entities
    
    # Add sampled entities
    for entity in sampled_entities:
        sampled_graph.add_entity(entity)
    
    sampled_entity_ids = {entity.id for entity in sampled_entities}
    
    # Add relations between sampled entities
    relation_count = 0
    for relation in graph_rag.relations.values():
        if relation.source in sampled_entity_ids and relation.target in sampled_entity_ids:
            sampled_graph.add_relation(relation)
            relation_count += 1
    
    print(f"✓ Sampled {len(sampled_entities)} entities and {relation_count} relations")
    return sampled_graph

def analyze_entity_types(graph_rag):
    """Analyze entity types in the graph"""
    print(f"\nEntity Type Analysis:")
    print(f"=" * 30)
    
    entity_types = {}
    for entity in graph_rag.entities.values():
        entity_type = entity.type
        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
    
    # Show top 10 most common entity types
    sorted_types = sorted(entity_types.items(), key=lambda x: x[1], reverse=True)
    print(f"Top Entity Types:")
    for entity_type, count in sorted_types[:10]:
        print(f"  {entity_type}: {count}")
    
    return entity_types

def analyze_relationship_types(graph_rag):
    """Analyze relationship types in the graph"""
    print(f"\nRelationship Type Analysis:")
    print(f"=" * 30)
    
    relation_types = {}
    for relation in graph_rag.relations.values():
        relation_type = relation.relation_type
        relation_types[relation_type] = relation_types.get(relation_type, 0) + 1
    
    # Show top 10 most common relationship types
    sorted_types = sorted(relation_types.items(), key=lambda x: x[1], reverse=True)
    print(f"Top Relationship Types:")
    for relation_type, count in sorted_types[:10]:
        print(f"  {relation_type}: {count}")
    
    return relation_types

def show_sample_entities(graph_rag, n=5):
    """Show sample entities from the graph"""
    print(f"\nSample Entities ({n} examples):")
    print(f"=" * 30)
    
    sample_entities = list(graph_rag.entities.values())[:n]
    for i, entity in enumerate(sample_entities, 1):
        print(f"{i}. {entity.name} ({entity.type})")
        if entity.properties:
            for key, value in list(entity.properties.items())[:3]:  # Show first 3 properties
                print(f"   {key}: {value}")
        print()

if __name__ == "__main__":
    # Load and visualize the knowledge graph
    full_graph, sampled_graph = load_specific_knowledge_graph()
    
    if full_graph:
        # Analyze the graph
        analyze_entity_types(full_graph)
        analyze_relationship_types(full_graph)
        show_sample_entities(full_graph)
        
        print(f"\n" + "="*50)
        print(f"FILES CREATED:")
        print(f"- knowledge_graph_3d.png (3D visualization of sampled graph)")
        print(f"- knowledge_graph_2d.png (2D visualization of sampled graph)")
        print(f"\nThe full graph has {len(full_graph.entities)} entities and {len(full_graph.relations)} relations.")
        print(f"Visualizations show a sample of {len(sampled_graph.entities)} entities for clarity.")
        print(f"✓ Knowledge graph successfully loaded and visualized in 3D!")
    else:
        print("✗ Failed to load knowledge graph.") 