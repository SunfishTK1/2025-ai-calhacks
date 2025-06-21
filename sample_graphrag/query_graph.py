import os
import pickle
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Tuple
from openai import AzureOpenAI
from dotenv import load_dotenv
import json
from sklearn.metrics.pairwise import cosine_similarity

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

class GraphRAGQuery:
    def __init__(self, graph_path: str):
        """Initialize the query interface with a saved graph"""
        with open(graph_path, 'rb') as f:
            graph_data = pickle.load(f)
        
        self.graph = nx.node_link_graph(graph_data['graph'])
        self.embeddings = graph_data['embeddings']
        self.chunks = graph_data['chunks']
        self.entity_map = graph_data['entity_map']
        self.relationship_map = graph_data['relationship_map']
        self.community_summaries = graph_data.get('community_summaries', {})
        
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for query text"""
        try:
            response = client.embeddings.create(
                model=embedding_deployment,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return list(np.random.rand(1536))
    
    def local_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Local search: Find entities most similar to the query and explore their neighborhood
        """
        print("Performing local search...")
        query_embedding = self.get_embedding(query)
        
        # Find most similar entities
        similarities = {}
        for entity_id, entity_embedding in self.embeddings.items():
            sim = cosine_similarity([query_embedding], [entity_embedding])[0][0]
            similarities[entity_id] = sim
        
        # Get top-k entities
        top_entities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Gather context from neighborhoods
        context_data = []
        for entity_id, score in top_entities:
            entity = self.entity_map.get(entity_id, {})
            
            # Get neighbors
            neighbors = list(self.graph.neighbors(entity_id))
            neighbor_info = []
            for neighbor in neighbors[:5]:  # Limit neighbors
                neighbor_entity = self.entity_map.get(neighbor, {})
                # Get edge data
                edge_data = self.graph.get_edge_data(entity_id, neighbor, {})
                neighbor_info.append({
                    "entity": neighbor_entity,
                    "relationship": edge_data
                })
            
            context_data.append({
                "entity": entity,
                "score": score,
                "neighbors": neighbor_info
            })
        
        # Generate response using context
        context_text = self._format_local_context(context_data)
        response = self._generate_response(query, context_text, "local")
        
        return {
            "type": "local_search",
            "query": query,
            "context": context_data,
            "response": response
        }
    
    def global_search(self, query: str) -> Dict[str, Any]:
        """
        Global search: Use community summaries to understand the overall structure
        """
        print("Performing global search...")
        query_embedding = self.get_embedding(query)
        
        # Find most relevant communities
        community_scores = {}
        for comm_id, comm_data in self.community_summaries.items():
            if 'embedding' in comm_data:
                sim = cosine_similarity([query_embedding], [comm_data['embedding']])[0][0]
                community_scores[comm_id] = sim
        
        # Get top communities
        top_communities = sorted(community_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Gather community information
        context_data = []
        for comm_id, score in top_communities:
            comm_data = self.community_summaries[comm_id]
            
            # Get sample entities from community
            sample_entities = []
            for node in comm_data['nodes'][:5]:
                entity = self.entity_map.get(node, {})
                sample_entities.append(entity)
            
            context_data.append({
                "community_id": comm_id,
                "summary": comm_data['summary'],
                "score": score,
                "sample_entities": sample_entities,
                "size": len(comm_data['nodes'])
            })
        
        # Generate response using global context
        context_text = self._format_global_context(context_data)
        response = self._generate_response(query, context_text, "global")
        
        return {
            "type": "global_search",
            "query": query,
            "context": context_data,
            "response": response
        }
    
    def hybrid_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Hybrid search: Combine local and global search strategies
        """
        print("Performing hybrid search...")
        
        # Perform both searches
        local_result = self.local_search(query, top_k)
        global_result = self.global_search(query)
        
        # Combine contexts
        combined_context = {
            "local_context": local_result['context'],
            "global_context": global_result['context']
        }
        
        # Generate comprehensive response
        context_text = f"""Local Context:
{self._format_local_context(local_result['context'])}

Global Context:
{self._format_global_context(global_result['context'])}"""
        
        response = self._generate_response(query, context_text, "hybrid")
        
        return {
            "type": "hybrid_search",
            "query": query,
            "context": combined_context,
            "response": response
        }
    
    def path_search(self, source_entity: str, target_entity: str) -> Dict[str, Any]:
        """
        Path search: Find paths between two entities
        """
        print(f"Finding paths between '{source_entity}' and '{target_entity}'...")
        
        # Find entities that match the names
        source_candidates = [e for e in self.entity_map if source_entity.lower() in e.lower()]
        target_candidates = [e for e in self.entity_map if target_entity.lower() in e.lower()]
        
        if not source_candidates or not target_candidates:
            return {
                "type": "path_search",
                "source": source_entity,
                "target": target_entity,
                "paths": [],
                "response": "Could not find one or both entities in the graph."
            }
        
        source = source_candidates[0]
        target = target_candidates[0]
        
        paths = []
        try:
            # Find shortest paths
            if nx.has_path(self.graph, source, target):
                # Get up to 3 shortest paths
                for path in list(nx.all_shortest_paths(self.graph, source, target))[:3]:
                    path_info = []
                    for i in range(len(path) - 1):
                        entity = self.entity_map.get(path[i], {})
                        edge_data = self.graph.get_edge_data(path[i], path[i+1], {})
                        path_info.append({
                            "entity": entity,
                            "relationship": edge_data
                        })
                    # Add last entity
                    path_info.append({"entity": self.entity_map.get(path[-1], {}), "relationship": None})
                    paths.append(path_info)
        except:
            pass
        
        # Generate response
        context_text = self._format_path_context(paths, source, target)
        response = self._generate_response(
            f"What connects {source_entity} to {target_entity}?", 
            context_text, 
            "path"
        )
        
        return {
            "type": "path_search",
            "source": source,
            "target": target,
            "paths": paths,
            "response": response
        }
    
    def _format_local_context(self, context_data: List[Dict]) -> str:
        """Format local search context for LLM"""
        context_parts = []
        for item in context_data:
            entity = item['entity']
            context_parts.append(f"Entity: {entity.get('name', 'Unknown')} ({entity.get('type', 'Unknown')})")
            context_parts.append(f"Description: {entity.get('description', 'No description')}")
            context_parts.append(f"Relevance Score: {item['score']:.3f}")
            
            if item['neighbors']:
                context_parts.append("Related to:")
                for neighbor in item['neighbors']:
                    n_entity = neighbor['entity']
                    rel = neighbor['relationship']
                    context_parts.append(f"  - {n_entity.get('name', 'Unknown')} via {rel.get('type', 'unknown relationship')}")
            
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _format_global_context(self, context_data: List[Dict]) -> str:
        """Format global search context for LLM"""
        context_parts = []
        for item in context_data:
            context_parts.append(f"Community (Size: {item['size']} entities):")
            context_parts.append(f"Summary: {item['summary']}")
            context_parts.append(f"Relevance Score: {item['score']:.3f}")
            context_parts.append("Sample entities:")
            for entity in item['sample_entities']:
                context_parts.append(f"  - {entity.get('name', 'Unknown')} ({entity.get('type', 'Unknown')})")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _format_path_context(self, paths: List[List[Dict]], source: str, target: str) -> str:
        """Format path search context for LLM"""
        if not paths:
            return f"No paths found between {source} and {target}"
        
        context_parts = [f"Paths from {source} to {target}:"]
        for i, path in enumerate(paths):
            context_parts.append(f"\nPath {i+1}:")
            for j, item in enumerate(path):
                entity = item['entity']
                context_parts.append(f"  {j+1}. {entity.get('name', 'Unknown')} ({entity.get('type', 'Unknown')})")
                if item['relationship']:
                    rel = item['relationship']
                    context_parts.append(f"     → {rel.get('type', 'relates to')} →")
        
        return "\n".join(context_parts)
    
    def _generate_response(self, query: str, context: str, search_type: str) -> str:
        """Generate response using LLM with context"""
        system_prompts = {
            "local": "You are an AI assistant that answers questions based on specific entities and their direct relationships from a knowledge graph.",
            "global": "You are an AI assistant that provides high-level insights based on community structures and patterns in a knowledge graph.",
            "hybrid": "You are an AI assistant that combines specific entity information with broader patterns to provide comprehensive answers.",
            "path": "You are an AI assistant that explains connections and relationships between entities in a knowledge graph."
        }
        
        prompt = f"""Based on the following context from the knowledge graph, answer the query.
        
Query: {query}

Context:
{context}

Provide a clear, informative answer based on the context provided."""
        
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": system_prompts.get(search_type, system_prompts["hybrid"])},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {e}"

def main():
    # Initialize query interface
    query_interface = GraphRAGQuery("knowledge_graph.pkl")
    
    print("GraphRAG Query Interface")
    print("Available search types:")
    print("1. Local Search - Find specific entities and their relationships")
    print("2. Global Search - Get high-level insights from community structures")
    print("3. Hybrid Search - Combine local and global perspectives")
    print("4. Path Search - Find connections between two entities")
    print("Type 'quit' to exit\n")
    
    while True:
        search_type = input("Select search type (1-4): ").strip()
        
        if search_type.lower() == 'quit':
            break
        
        if search_type == "1":
            query = input("Enter your query: ")
            result = query_interface.local_search(query)
            print(f"\nResponse: {result['response']}\n")
            
        elif search_type == "2":
            query = input("Enter your query: ")
            result = query_interface.global_search(query)
            print(f"\nResponse: {result['response']}\n")
            
        elif search_type == "3":
            query = input("Enter your query: ")
            result = query_interface.hybrid_search(query)
            print(f"\nResponse: {result['response']}\n")
            
        elif search_type == "4":
            source = input("Enter source entity: ")
            target = input("Enter target entity: ")
            result = query_interface.path_search(source, target)
            print(f"\nResponse: {result['response']}\n")
            
        else:
            print("Invalid option. Please select 1-4.\n")

if __name__ == "__main__":
    main()