#!/usr/bin/env python3
"""
Graph RAG Implementation
========================
A Graph-based Retrieval-Augmented Generation system with 3D visualization
"""

import json
import pickle
import networkx as nx
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@dataclass
class Entity:
    id: str
    name: str
    type: str
    properties: Dict[str, Any]


@dataclass  
class Relation:
    id: str
    source: str
    target: str
    relation_type: str
    properties: Dict[str, Any]


class GraphRAG:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entities = {}
        self.relations = {}
        self.documents = {}
    
    def add_entity(self, entity: Entity):
        self.entities[entity.id] = entity
        self.graph.add_node(entity.id, name=entity.name, type=entity.type)
        print(f"Added entity: {entity.name}")
    
    def add_relation(self, relation: Relation):
        self.relations[relation.id] = relation
        self.graph.add_edge(relation.source, relation.target, 
                           relation_type=relation.relation_type)
        print(f"Added relation: {relation.relation_type}")
    
    def query(self, entity_name: str):
        """Query the graph for information about an entity"""
        matches = []
        for eid, entity in self.entities.items():
            if entity_name.lower() in entity.name.lower():
                matches.append(entity)
        return matches
    
    def visualize_2d(self, save_path=None):
        """Visualize the knowledge graph in 2D"""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        
        nx.draw_networkx_nodes(self.graph, pos, node_size=500, 
                              node_color='lightblue', alpha=0.8)
        nx.draw_networkx_edges(self.graph, pos, alpha=0.6)
        
        labels = {node: self.entities[node].name for node in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
        plt.title("Knowledge Graph (2D)")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_3d(self, save_path=None):
        """Visualize the knowledge graph in 3D"""
        if len(self.graph.nodes()) == 0:
            print("No nodes to visualize")
            return
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get 3D positions using NetworkX layout and adding Z dimension
        pos_2d = nx.spring_layout(self.graph, k=3, iterations=50)
        
        # Convert to 3D positions
        pos_3d = {}
        node_colors = []
        node_sizes = []
        
        # Define colors for different entity types
        type_colors = {
            'PERSON': 'red',
            'ORGANIZATION': 'blue', 
            'LOCATION': 'green',
            'CONCEPT': 'orange',
            'EVENT': 'purple'
        }
        
        for i, (node, (x, y)) in enumerate(pos_2d.items()):
            # Add some randomness to Z coordinate for better visualization
            z = np.random.uniform(-1, 1) if len(self.graph.nodes()) > 1 else 0
            pos_3d[node] = (x, y, z)
            
            # Get entity type for coloring
            entity_type = self.entities.get(node, Entity('', '', 'UNKNOWN', {})).type
            color = type_colors.get(entity_type, 'gray')
            node_colors.append(color)
            
            # Node size based on degree
            degree = self.graph.degree(node)
            node_sizes.append(max(100, degree * 200))
        
        # Extract coordinates
        xs = [pos_3d[node][0] for node in self.graph.nodes()]
        ys = [pos_3d[node][1] for node in self.graph.nodes()]
        zs = [pos_3d[node][2] for node in self.graph.nodes()]
        
        # Plot nodes
        scatter = ax.scatter(xs, ys, zs, c=node_colors, s=node_sizes, alpha=0.8, edgecolors='black')
        
        # Plot edges
        for edge in self.graph.edges():
            x_coords = [pos_3d[edge[0]][0], pos_3d[edge[1]][0]]
            y_coords = [pos_3d[edge[0]][1], pos_3d[edge[1]][1]]
            z_coords = [pos_3d[edge[0]][2], pos_3d[edge[1]][2]]
            ax.plot(x_coords, y_coords, z_coords, 'gray', alpha=0.6, linewidth=1)
        
        # Add node labels
        for node in self.graph.nodes():
            if node in self.entities:
                ax.text(pos_3d[node][0], pos_3d[node][1], pos_3d[node][2] + 0.1, 
                       self.entities[node].name, fontsize=8, ha='center')
        
        # Customize the plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Knowledge Graph (3D Visualization)', fontsize=14, pad=20)
        
        # Create legend for entity types
        legend_elements = []
        for entity_type, color in type_colors.items():
            if any(self.entities.get(node, Entity('', '', 'UNKNOWN', {})).type == entity_type 
                   for node in self.graph.nodes()):
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=color, markersize=10, 
                                                label=entity_type))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
        
        # Improve the viewing angle
        ax.view_init(elev=20, azim=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D visualization saved to {save_path}")
        
        plt.show()
    
    def save_to_file(self, filepath: str):
        """Save the graph to a JSON file"""
        data = {
            'entities': {k: {'id': v.id, 'name': v.name, 'type': v.type, 
                           'properties': v.properties} for k, v in self.entities.items()},
            'relations': {k: {'id': v.id, 'source': v.source, 'target': v.target,
                            'relation_type': v.relation_type, 'properties': v.properties} 
                         for k, v in self.relations.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Graph saved to {filepath}")
    
    def save_to_pickle(self, filepath: str):
        """Save the graph to a pickle file"""
        data = {
            'graph': self.graph,
            'entities': self.entities,
            'relations': self.relations,
            'documents': self.documents
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Graph saved to pickle file: {filepath}")
    
    def load_from_pickle(self, filepath: str):
        """Load the graph from a pickle file"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.graph = data.get('graph', nx.MultiDiGraph())
            self.entities = data.get('entities', {})
            self.relations = data.get('relations', {})
            self.documents = data.get('documents', {})
            
            print(f"Graph loaded from pickle file: {filepath}")
            print(f"Loaded {len(self.entities)} entities and {len(self.relations)} relations")
            
        except FileNotFoundError:
            print(f"File not found: {filepath}")
        except Exception as e:
            print(f"Error loading pickle file: {e}")
    
    def load_networkx_pickle(self, filepath: str):
        """Load a NetworkX graph from pickle and convert to GraphRAG format"""
        try:
            # Load the NetworkX graph
            nx_graph = nx.read_gpickle(filepath)
            print(f"Loaded NetworkX graph with {nx_graph.number_of_nodes()} nodes and {nx_graph.number_of_edges()} edges")
            
            # Clear existing data
            self.graph = nx.MultiDiGraph()
            self.entities = {}
            self.relations = {}
            
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
                self.add_entity(entity)
            
            # Convert edges to relations
            for i, edge in enumerate(nx_graph.edges(data=True)):
                source, target = str(edge[0]), str(edge[1])
                edge_data = edge[2] if len(edge) > 2 else {}
                
                relation = Relation(
                    id=f"rel_{i}",
                    source=source,
                    target=target,
                    relation_type=edge_data.get('relation_type', 'CONNECTED'),
                    properties=edge_data
                )
                self.add_relation(relation)
                
        except FileNotFoundError:
            print(f"File not found: {filepath}")
        except Exception as e:
            print(f"Error loading NetworkX pickle: {e}")
    
    def get_statistics(self):
        """Get graph statistics"""
        return {
            'nodes': len(self.graph.nodes()),
            'edges': len(self.graph.edges()),
            'entities': len(self.entities),
            'relations': len(self.relations),
            'density': nx.density(self.graph) if len(self.graph.nodes()) > 1 else 0,
            'is_connected': nx.is_connected(self.graph.to_undirected()) if len(self.graph.nodes()) > 1 else True
        }


def create_sample_graph():
    """Create a more complex sample graph for 3D visualization"""
    graph_rag = GraphRAG()
    
    # Add entities
    entities = [
        Entity("john_smith", "John Smith", "PERSON", {"age": 30, "department": "Engineering"}),
        Entity("alice_johnson", "Alice Johnson", "PERSON", {"age": 28, "department": "Marketing"}),
        Entity("bob_wilson", "Bob Wilson", "PERSON", {"age": 35, "department": "Sales"}),
        Entity("tech_corp", "Tech Corp", "ORGANIZATION", {"industry": "Technology", "employees": 500}),
        Entity("data_corp", "Data Corp", "ORGANIZATION", {"industry": "Data Analytics", "employees": 200}),
        Entity("san_francisco", "San Francisco", "LOCATION", {"state": "California", "population": 875000}),
        Entity("new_york", "New York", "LOCATION", {"state": "New York", "population": 8400000}),
        Entity("ai_project", "AI Development Project", "CONCEPT", {"status": "active", "budget": 1000000}),
        Entity("data_analysis", "Data Analysis", "CONCEPT", {"status": "completed", "duration": "6 months"}),
        Entity("conference_2024", "Tech Conference 2024", "EVENT", {"date": "2024-03-15", "attendees": 1000})
    ]
    
    for entity in entities:
        graph_rag.add_entity(entity)
    
    # Add relations
    relations = [
        Relation("rel_1", "john_smith", "tech_corp", "WORKS_AT", {"role": "Senior Engineer"}),
        Relation("rel_2", "alice_johnson", "tech_corp", "WORKS_AT", {"role": "Marketing Manager"}),
        Relation("rel_3", "bob_wilson", "data_corp", "WORKS_AT", {"role": "Sales Director"}),
        Relation("rel_4", "tech_corp", "san_francisco", "LOCATED_IN", {}),
        Relation("rel_5", "data_corp", "new_york", "LOCATED_IN", {}),
        Relation("rel_6", "john_smith", "ai_project", "WORKS_ON", {"contribution": "lead developer"}),
        Relation("rel_7", "alice_johnson", "ai_project", "MANAGES", {"responsibility": "project coordination"}),
        Relation("rel_8", "bob_wilson", "data_analysis", "COMPLETED", {"role": "client liaison"}),
        Relation("rel_9", "tech_corp", "data_corp", "PARTNERS_WITH", {"partnership_type": "strategic"}),
        Relation("rel_10", "john_smith", "conference_2024", "PRESENTS_AT", {"topic": "AI in Enterprise"}),
        Relation("rel_11", "alice_johnson", "conference_2024", "ATTENDS", {"purpose": "networking"}),
        Relation("rel_12", "ai_project", "data_analysis", "BUILDS_ON", {"dependency": "uses insights from"})
    ]
    
    for relation in relations:
        graph_rag.add_relation(relation)
    
    return graph_rag


def demo():
    """Demo of the GraphRAG system with 3D visualization"""
    print("GraphRAG 3D Demo")
    print("================")
    
    # Create a complex sample graph
    print("\nCreating sample knowledge graph...")
    graph_rag = create_sample_graph()
    
    # Show statistics
    stats = graph_rag.get_statistics()
    print(f"\nGraph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Query example
    print("\nQuerying for 'John':")
    results = graph_rag.query("John")
    for result in results:
        print(f"Found: {result.name} ({result.type})")
    
    # Save to different formats
    print("\nSaving graph...")
    graph_rag.save_to_file("complex_graph_data.json")
    graph_rag.save_to_pickle("complex_graph.pkl")
    
    # Demonstrate loading from pickle
    print("\nTesting pickle load...")
    new_graph_rag = GraphRAG()
    new_graph_rag.load_from_pickle("complex_graph.pkl")
    
    # 3D Visualization
    print("\nGenerating 3D visualization...")
    new_graph_rag.visualize_3d("graph_3d_visualization.png")
    
    # Also show 2D for comparison
    print("\nGenerating 2D visualization for comparison...")
    new_graph_rag.visualize_2d("graph_2d_visualization.png")
    
    return new_graph_rag


def demo_load_external_pickle():
    """Demo loading an external NetworkX pickle file"""
    print("\nDemo: Loading External NetworkX Pickle")
    print("======================================")
    
    # Create a sample NetworkX graph and save it
    print("Creating sample NetworkX graph...")
    G = nx.karate_club_graph()  # Famous social network graph
    
    # Add some attributes for better visualization
    for node in G.nodes():
        G.nodes[node]['name'] = f'Person_{node}'
        G.nodes[node]['type'] = 'PERSON'
        G.nodes[node]['club'] = G.nodes[node].get('club', 'unknown')
    
    # Save as pickle
    nx.write_gpickle(G, "karate_club.pkl")
    print("Saved sample NetworkX graph to karate_club.pkl")
    
    # Load into GraphRAG
    graph_rag = GraphRAG()
    graph_rag.load_networkx_pickle("karate_club.pkl")
    
    # Show statistics
    stats = graph_rag.get_statistics()
    print(f"\nLoaded Graph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Visualize in 3D
    print("\nVisualizing loaded graph in 3D...")
    graph_rag.visualize_3d("karate_club_3d.png")
    
    return graph_rag


if __name__ == "__main__":
    # Run the main demo
    main_graph = demo()
    
    print("\n" + "="*50)
    
    # Run the external pickle demo
    karate_graph = demo_load_external_pickle()
    
    print(f"\nDemo completed!")
    print(f"Files created:")
    print(f"- complex_graph.pkl (GraphRAG pickle file)")
    print(f"- karate_club.pkl (NetworkX pickle file)")
    print(f"- graph_3d_visualization.png (3D plot)")
    print(f"- graph_2d_visualization.png (2D plot)")
    print(f"- karate_club_3d.png (Karate club 3D plot)") 