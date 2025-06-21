import os
import pandas as pd
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Any
import pickle
import json
from openai import AzureOpenAI
from dotenv import load_dotenv
import re
from collections import defaultdict
import time

load_dotenv()

# Configuration
endpoint = os.getenv("ENDPOINT_URL", "https://2025-ai-hackberkeley.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "o4-mini")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
embedding_deployment = os.getenv("EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large")

if not subscription_key:
    raise ValueError("Please set the AZURE_OPENAI_API_KEY environment variable")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2025-01-01-preview",
)

class GraphRAGBuilder:
    def __init__(self, chunk_size=1000):
        self.graph = nx.Graph()
        self.embeddings = {}
        self.chunks = []
        self.chunk_size = chunk_size
        self.entity_map = {}
        self.relationship_map = defaultdict(list)
        
    def csv_to_text(self, csv_path: str) -> str:
        """Convert CSV file to text format"""
        df = pd.read_csv(csv_path)
        
        # Convert each row to a text description
        text_chunks = []
        for idx, row in df.iterrows():
            row_text = f"Record {idx + 1}: "
            row_items = []
            for col, val in row.items():
                if pd.notna(val):
                    row_items.append(f"{col}={val}")
            row_text += ", ".join(row_items)
            text_chunks.append(row_text)
        
        return "\n".join(text_chunks)
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for processing"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            
        return chunks
    
    def extract_entities_relationships(self, text_chunk: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities and relationships from text using LLM"""
        prompt = f"""Extract entities and their relationships from the following text. 
        Return the result in JSON format with two lists:
        1. "entities": list of objects with "name", "type", and "description"
        2. "relationships": list of objects with "source", "target", "type", and "description"
        
        Text: {text_chunk}
        
        Example format:
        {{
            "entities": [
                {{"name": "John Smith", "type": "Person", "description": "CEO of Company X"}},
                {{"name": "Company X", "type": "Organization", "description": "Tech company"}}
            ],
            "relationships": [
                {{"source": "John Smith", "target": "Company X", "type": "works_for", "description": "CEO position"}}
            ]
        }}
        """
        
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured information from text."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=90000,
            )
            
            result = response.choices[0].message.content
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return parsed.get("entities", []), parsed.get("relationships", [])
            
        except Exception as e:
            print(f"Error extracting entities: {e}")
            
        return [], []
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Azure OpenAI"""
        try:
            response = client.embeddings.create(
                model=embedding_deployment,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Return random embedding as fallback
            return list(np.random.rand(1536))
    
    def build_graph_from_csv(self, csv_path: str):
        """Build knowledge graph from CSV file"""
        print("Converting CSV to text...")
        text = self.csv_to_text(csv_path)
        
        print("Chunking text...")
        self.chunks = self.chunk_text(text)
        
        print(f"Processing {len(self.chunks)} chunks...")
        all_entities = []
        all_relationships = []
        
        for i, chunk in enumerate(self.chunks):
            print(f"Processing chunk {i+1}/{len(self.chunks)}...")
            entities, relationships = self.extract_entities_relationships(chunk)
            all_entities.extend(entities)
            all_relationships.extend(relationships)
            
            # Add small delay to avoid rate limiting
            time.sleep(0.5)
        
        # Build graph
        print("Building graph structure...")
        for entity in all_entities:
            entity_id = entity['name']
            self.graph.add_node(entity_id, **entity)
            self.entity_map[entity_id] = entity
            
            # Get embedding for entity
            entity_text = f"{entity['name']} ({entity['type']}): {entity.get('description', '')}"
            self.embeddings[entity_id] = self.get_embedding(entity_text)
        
        for rel in all_relationships:
            if rel['source'] in self.graph and rel['target'] in self.graph:
                self.graph.add_edge(rel['source'], rel['target'], **rel)
                self.relationship_map[rel['source']].append(rel)
                self.relationship_map[rel['target']].append(rel)
        
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def create_community_summaries(self):
        """Create summaries for communities in the graph"""
        # Detect communities using Louvain method
        try:
            import community as community_louvain
            communities = community_louvain.best_partition(self.graph)
        except:
            # Fallback to simple connected components
            communities = {}
            for i, component in enumerate(nx.connected_components(self.graph)):
                for node in component:
                    communities[node] = i
        
        # Group nodes by community
        community_groups = defaultdict(list)
        for node, comm_id in communities.items():
            community_groups[comm_id].append(node)
        
        # Create summaries for each community
        self.community_summaries = {}
        for comm_id, nodes in community_groups.items():
            if len(nodes) > 1:
                # Get entities in this community
                entities_desc = []
                for node in nodes[:10]:  # Limit to first 10 nodes
                    entity = self.entity_map.get(node, {})
                    entities_desc.append(f"{entity.get('name', node)} ({entity.get('type', 'Unknown')})")
                
                summary_prompt = f"""Create a brief summary of this group of related entities:
                {', '.join(entities_desc)}
                
                Provide a 1-2 sentence summary of what connects these entities."""
                
                try:
                    response = client.chat.completions.create(
                        model=deployment,
                        messages=[
                            {"role": "system", "content": "You are an expert at summarizing relationships between entities."},
                            {"role": "user", "content": summary_prompt}
                        ],
                        max_completion_tokens=90000,
                    )
                    
                    self.community_summaries[comm_id] = {
                        "summary": response.choices[0].message.content,
                        "nodes": nodes,
                        "embedding": self.get_embedding(response.choices[0].message.content)
                    }
                except Exception as e:
                    print(f"Error creating community summary: {e}")
    
    def save_graph(self, output_path: str):
        """Save the graph and associated data"""
        graph_data = {
            "graph": nx.node_link_data(self.graph),
            "embeddings": self.embeddings,
            "chunks": self.chunks,
            "entity_map": self.entity_map,
            "relationship_map": dict(self.relationship_map),
            "community_summaries": getattr(self, 'community_summaries', {})
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(graph_data, f)
        
        print(f"Graph saved to {output_path}")
        
        # Also save a JSON summary for inspection
        summary = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_chunks": len(self.chunks),
            "num_communities": len(getattr(self, 'community_summaries', {})),
            "sample_entities": list(self.entity_map.keys())[:10]
        }
        
        with open(output_path.replace('.pkl', '_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

def main():
    # Example usage
    builder = GraphRAGBuilder(chunk_size=500)
    
    # Replace with your CSV file path
    csv_path = "biz.csv"
    
    if os.path.exists(csv_path):
        builder.build_graph_from_csv(csv_path)
        builder.create_community_summaries()
        builder.save_graph("knowledge_graph.pkl")
    else:
        print(f"CSV file not found: {csv_path}")
        print("Please provide a valid CSV file path.")

if __name__ == "__main__":
    main()