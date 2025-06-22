#!/usr/bin/env python3
"""
Load and Visualize knowledge_graph.pkl in 3D
============================================
This script loads the knowledge_graph.pkl file and creates a 3D visualization
"""

from graphrag import GraphRAG, Entity, Relation
import matplotlib.pyplot as plt
import numpy as np
import pickle
import networkx as nx

def inspect_pickle_file(filepath):
    """Inspect the structure of a pickle file"""
    print(f"Inspecting {filepath}...")
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Pickle file type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Dictionary keys: {list(data.keys())}")
            for key, value in data.items():
                print(f"  {key}: {type(value)} - {len(value) if hasattr(value, '__len__') else 'N/A'}")
                if hasattr(value, '__len__') and len(value) > 0:
                    if isinstance(value, dict):
                        sample_key = list(value.keys())[0]
                        print(f"    Sample key: {sample_key} -> {type(value[sample_key])}")
                    elif isinstance(value, list):
                        print(f"    Sample item: {type(value[0])}")
        
        elif hasattr(data, 'nodes'):
            print(f"NetworkX graph detected: {data.number_of_nodes()} nodes, {data.number_of_edges()} edges")
        
        return data
    except Exception as e:
        print(f"Error inspecting pickle file: {e}")
        return None

def convert_dict_to_graphrag(data):
    """Convert dictionary data to GraphRAG format"""
    graph_rag = GraphRAG()
    
    try:
        # Try to find entities and relations in the dictionary
        entities_data = None
        relations_data = None
        graph_data = None
        
        # Look for common keys
        for key in ['entities', 'nodes', 'vertices']:
            if key in data:
                entities_data = data[key]
                break
        
        for key in ['relations', 'edges', 'relationships', 'links']:
            if key in data:
                relations_data = data[key]
                break
        
        for key in ['graph', 'network', 'nx_graph']:
            if key in data:
                graph_data = data[key]
                break
        
        # If we have a NetworkX graph, use it
        if graph_data and hasattr(graph_data, 'nodes'):
            print("Found NetworkX graph in data")
            convert_networkx_to_graphrag(graph_data, graph_rag)
            return graph_rag
        
        # Convert entities
        if entities_data:
            print(f"Converting {len(entities_data) if hasattr(entities_data, '__len__') else 'unknown'} entities...")
            
            if isinstance(entities_data, dict):
                for entity_id, entity_info in entities_data.items():
                    if isinstance(entity_info, dict):
                        entity = Entity(
                            id=str(entity_id),
                            name=entity_info.get('name', str(entity_id)),
                            type=entity_info.get('type', 'UNKNOWN'),
                            properties=entity_info
                        )
                        graph_rag.add_entity(entity)
            elif isinstance(entities_data, list):
                for i, entity_info in enumerate(entities_data):
                    if isinstance(entity_info, dict):
                        entity = Entity(
                            id=entity_info.get('id', f'entity_{i}'),
                            name=entity_info.get('name', f'Entity_{i}'),
                            type=entity_info.get('type', 'UNKNOWN'),
                            properties=entity_info
                        )
                        graph_rag.add_entity(entity)
        
        # Convert relations
        if relations_data:
            print(f"Converting {len(relations_data) if hasattr(relations_data, '__len__') else 'unknown'} relations...")
            
            if isinstance(relations_data, dict):
                for relation_id, relation_info in relations_data.items():
                    if isinstance(relation_info, dict):
                        relation = Relation(
                            id=str(relation_id),
                            source=str(relation_info.get('source', relation_info.get('from', ''))),
                            target=str(relation_info.get('target', relation_info.get('to', ''))),
                            relation_type=relation_info.get('type', relation_info.get('relation_type', 'CONNECTED')),
                            properties=relation_info
                        )
                        graph_rag.add_relation(relation)
            elif isinstance(relations_data, list):
                for i, relation_info in enumerate(relations_data):
                    if isinstance(relation_info, dict):
                        relation = Relation(
                            id=relation_info.get('id', f'relation_{i}'),
                            source=str(relation_info.get('source', relation_info.get('from', ''))),
                            target=str(relation_info.get('target', relation_info.get('to', ''))),
                            relation_type=relation_info.get('type', relation_info.get('relation_type', 'CONNECTED')),
                            properties=relation_info
                        )
                        graph_rag.add_relation(relation)
        
        return graph_rag
        
    except Exception as e:
        print(f"Error converting dictionary to GraphRAG: {e}")
        return None

def convert_networkx_to_graphrag(nx_graph, graph_rag):
    """Convert NetworkX graph to GraphRAG format"""
    # Convert nodes to entities
    for node in nx_graph.nodes(data=True):
        node_id = str(node[0])
        node_data = node[1] if len(node) > 1 else {}
        
        entity = Entity(
            id=node_id,
            name=node_data.get('name', f'Node_{node_id}'),
            type=node_data.get('type', 'UNKNOWN'),
            properties=node_data
        )
        graph_rag.add_entity(entity)
    
    # Convert edges to relations
    for i, edge in enumerate(nx_graph.edges(data=True)):
        source, target = str(edge[0]), str(edge[1])
        edge_data = edge[2] if len(edge) > 2 else {}
        
        relation = Relation(
            id=f"rel_{i}",
            source=source,
            target=target,
            relation_type=edge_data.get('relation_type', edge_data.get('type', 'CONNECTED')),
            properties=edge_data
        )
        graph_rag.add_relation(relation)

def load_and_visualize_knowledge_graph():
    """Load knowledge_graph.pkl and create 3D visualization"""
    print("Loading knowledge_graph.pkl for 3D Visualization")
    print("=" * 50)
    
    # First, inspect the pickle file
    data = inspect_pickle_file("knowledge_graph.pkl")
    if data is None:
        return None
    
    # Initialize GraphRAG
    graph_rag = GraphRAG()
    
    # Try different loading strategies
    print("\nTrying to convert data to GraphRAG format...")
    
    if isinstance(data, dict):
        # Handle dictionary format
        if 'graph' in data and hasattr(data['graph'], 'nodes'):
            # GraphRAG format with NetworkX graph
            graph_rag.graph = data['graph']
            graph_rag.entities = data.get('entities', {})
            graph_rag.relations = data.get('relations', {})
            graph_rag.documents = data.get('documents', {})
        else:
            # Try to convert dictionary structure
            graph_rag = convert_dict_to_graphrag(data)
            
    elif hasattr(data, 'nodes'):
        # NetworkX graph
        convert_networkx_to_graphrag(data, graph_rag)
    
    else:
        print(f"Unknown data format: {type(data)}")
        return None
    
    if graph_rag is None:
        print("Failed to convert data to GraphRAG format")
        return None
    
    # Show statistics
    try:
        stats = graph_rag.get_statistics()
        print(f"\nKnowledge Graph Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error getting statistics: {e}")
        return None
    
    # If the graph is very large, we might want to sample it for visualization
    if stats['nodes'] > 1000:
        print(f"\nGraph is large ({stats['nodes']} nodes). Sampling for better visualization.")
        graph_rag = sample_large_graph(graph_rag, max_nodes=500)
        stats = graph_rag.get_statistics()
        print(f"\nSampled Graph Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # Create 3D visualization
    print(f"\nGenerating 3D visualization...")
    try:
        graph_rag.visualize_3d("knowledge_graph_3d.png")
        print("3D visualization completed!")
        
        # Also create 2D for comparison
        print("Generating 2D visualization for comparison...")
        graph_rag.visualize_2d("knowledge_graph_2d.png")
        print("2D visualization completed!")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        print("This might be due to the graph size or structure.")
    
    return graph_rag

def sample_large_graph(graph_rag, max_nodes=500):
    """Sample a large graph to make it more manageable for visualization"""
    print(f"Sampling graph to {max_nodes} nodes...")
    
    import random
    
    # Create a new GraphRAG with sampled nodes
    sampled_graph = GraphRAG()
    
    # Sample nodes
    all_nodes = list(graph_rag.graph.nodes())
    if len(all_nodes) > max_nodes:
        sampled_nodes = random.sample(all_nodes, max_nodes)
    else:
        sampled_nodes = all_nodes
    
    # Add sampled entities
    for node_id in sampled_nodes:
        if node_id in graph_rag.entities:
            sampled_graph.add_entity(graph_rag.entities[node_id])
        else:
            # Create a basic entity if not found
            entity = Entity(
                id=node_id,
                name=f"Node_{node_id}",
                type="UNKNOWN",
                properties={}
            )
            sampled_graph.add_entity(entity)
    
    # Add relations between sampled nodes
    relation_count = 0
    for relation in graph_rag.relations.values():
        if relation.source in sampled_nodes and relation.target in sampled_nodes:
            sampled_graph.add_relation(relation)
            relation_count += 1
    
    # If no relations found, try to get edges from the graph directly
    if relation_count == 0:
        for edge in graph_rag.graph.edges():
            if edge[0] in sampled_nodes and edge[1] in sampled_nodes:
                relation = Relation(
                    id=f"sampled_rel_{relation_count}",
                    source=edge[0],
                    target=edge[1],
                    relation_type="CONNECTED",
                    properties={}
                )
                sampled_graph.add_relation(relation)
                relation_count += 1
    
    print(f"Sampled {len(sampled_nodes)} nodes and {relation_count} relations")
    return sampled_graph

def analyze_graph_structure(graph_rag):
    """Analyze the structure of the loaded graph"""
    print(f"\nGraph Structure Analysis:")
    print(f"=" * 30)
    
    # Entity type distribution
    entity_types = {}
    for entity in graph_rag.entities.values():
        entity_types[entity.type] = entity_types.get(entity.type, 0) + 1
    
    print(f"Entity Types:")
    for entity_type, count in sorted(entity_types.items()):
        print(f"  {entity_type}: {count}")
    
    # Relation type distribution
    relation_types = {}
    for relation in graph_rag.relations.values():
        relation_types[relation.relation_type] = relation_types.get(relation.relation_type, 0) + 1
    
    print(f"\nRelation Types:")
    for relation_type, count in sorted(relation_types.items()):
        print(f"  {relation_type}: {count}")
    
    # Degree distribution
    degrees = [graph_rag.graph.degree(node) for node in graph_rag.graph.nodes()]
    if degrees:
        print(f"\nDegree Statistics:")
        print(f"  Min degree: {min(degrees)}")
        print(f"  Max degree: {max(degrees)}")
        print(f"  Average degree: {np.mean(degrees):.2f}")
        print(f"  Median degree: {np.median(degrees):.2f}")

def interactive_query(graph_rag):
    """Interactive query interface"""
    print(f"\nInteractive Query Interface")
    print(f"=" * 30)
    print("Enter entity names to search for (or 'quit' to exit):")
    
    while True:
        query = input("\nQuery: ").strip()
        if query.lower() == 'quit':
            break
        
        if query:
            results = graph_rag.query(query)
            if results:
                print(f"Found {len(results)} matching entities:")
                for i, result in enumerate(results[:10]):  # Show top 10
                    print(f"  {i+1}. {result.name} ({result.type})")
                if len(results) > 10:
                    print(f"  ... and {len(results) - 10} more")
            else:
                print("No matching entities found.")

if __name__ == "__main__":
    # Load and visualize the knowledge graph
    graph_rag = load_and_visualize_knowledge_graph()
    
    if graph_rag:
        # Analyze graph structure
        analyze_graph_structure(graph_rag)
        
        # Interactive query interface
        print(f"\nWould you like to try interactive queries? (y/n): ", end="")
        response = input().lower().strip()
        if response == 'y':
            interactive_query(graph_rag)
        
        print(f"\nFiles created:")
        print(f"- knowledge_graph_3d.png (3D visualization)")
        print(f"- knowledge_graph_2d.png (2D visualization)")
        print(f"\nVisualization complete!")
    else:
        print("Failed to load knowledge graph.") 