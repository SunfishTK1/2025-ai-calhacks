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
import logging
from datetime import datetime
import concurrent.futures
import threading
from collections import deque

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('graphrag_build.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("=== GraphRAG Builder Starting ===")

# Rate limiting setup
class RateLimiter:
    def __init__(self, max_requests_per_minute=700):
        self.max_requests = max_requests_per_minute
        self.requests = deque()
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            while self.requests and now - self.requests[0] > 60:
                self.requests.popleft()
            
            # If we're at the limit, wait
            if len(self.requests) >= self.max_requests:
                sleep_time = 60 - (now - self.requests[0]) + 0.1  # Small buffer
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                # Clean up again after sleeping
                now = time.time()
                while self.requests and now - self.requests[0] > 60:
                    self.requests.popleft()
            
            # Record this request
            self.requests.append(now)

rate_limiter = RateLimiter(max_requests_per_minute=700)

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
        logger.info(f"Initializing GraphRAGBuilder with chunk_size={chunk_size}")
        self.graph = nx.Graph()
        self.embeddings = {}
        self.chunks = []
        self.chunk_size = chunk_size
        self.entity_map = {}
        self.relationship_map = defaultdict(list)
        
        # Initialize output file for model responses
        self.output_file = f"model_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(self.output_file, 'w') as f:
            f.write(f"GraphRAG Model Outputs - {datetime.now()}\n")
            f.write("=" * 50 + "\n\n")
        
        # Thread lock for file writing
        self.file_lock = threading.Lock()
        
        logger.info(f"GraphRAGBuilder initialized successfully. Model outputs will be saved to: {self.output_file}")
        
    def csv_to_text(self, csv_path: str) -> str:
        """Convert CSV file to text format"""
        logger.info(f"Starting CSV to text conversion for: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            logger.info(f"CSV columns: {list(df.columns)}")
        except Exception as e:
            logger.error(f"Failed to load CSV file: {e}")
            raise
        
        # Convert each row to a text description
        text_chunks = []
        processed_rows = 0
        
        for idx, row in df.iterrows():
            row_text = f"Record {idx + 1}: "
            row_items = []
            non_null_values = 0
            
            for col, val in row.items():
                if pd.notna(val):
                    row_items.append(f"{col}={val}")
                    non_null_values += 1
            
            row_text += ", ".join(row_items)
            text_chunks.append(row_text)
            processed_rows += 1
            
            if processed_rows % 100 == 0:
                logger.info(f"Processed {processed_rows}/{len(df)} rows")
        
        combined_text = "\n".join(text_chunks)
        logger.info(f"CSV conversion complete. Generated text with {len(combined_text)} characters")
        return combined_text
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for processing"""
        logger.info(f"Starting text chunking with chunk_size={self.chunk_size}")
        words = text.split()
        logger.info(f"Total words to chunk: {len(words)}")
        chunks = []
        
        for i in range(0, len(words), self.chunk_size):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def extract_entities_relationships(self, text_chunk: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities and relationships from text using LLM"""
        logger.debug(f"Extracting entities from chunk of {len(text_chunk)} characters")
        
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
            logger.debug("Sending request to Azure OpenAI for entity extraction")
            
            # Apply rate limiting before making the request
            rate_limiter.wait_if_needed()
            
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured information from text."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=90000,
            )
            
            result = response.choices[0].message.content
            logger.debug(f"Received response of {len(result)} characters")
            
            # Save raw model output to file
            with self.file_lock:
                with open(self.output_file, 'a') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"CHUNK EXTRACTION - {datetime.now()}\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"Input chunk ({len(text_chunk)} chars):\n")
                    f.write(f"{text_chunk[:500]}{'...' if len(text_chunk) > 500 else ''}\n\n")
                    f.write(f"Model Response:\n")
                    f.write(f"{result}\n")
                    f.write(f"{'='*60}\n\n")
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                entities = parsed.get("entities", [])
                relationships = parsed.get("relationships", [])
                
                # Log extraction details to file
                with self.file_lock:
                    with open(self.output_file, 'a') as f:
                        f.write(f"EXTRACTION RESULTS:\n")
                        f.write(f"Entities extracted: {len(entities)}\n")
                        for i, entity in enumerate(entities[:10]):  # First 10 entities
                            f.write(f"  {i+1}. {entity.get('name', 'N/A')} ({entity.get('type', 'N/A')}): {entity.get('description', 'N/A')}\n")
                        if len(entities) > 10:
                            f.write(f"  ... and {len(entities) - 10} more entities\n")
                        
                        f.write(f"\nRelationships extracted: {len(relationships)}\n")
                        for i, rel in enumerate(relationships[:10]):  # First 10 relationships
                            f.write(f"  {i+1}. {rel.get('source', 'N/A')} -> {rel.get('target', 'N/A')} ({rel.get('type', 'N/A')})\n")
                        if len(relationships) > 10:
                            f.write(f"  ... and {len(relationships) - 10} more relationships\n")
                        f.write(f"\n")
                
                logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
                return entities, relationships
            else:
                logger.warning("No JSON found in response")
                with self.file_lock:
                    with open(self.output_file, 'a') as f:
                        f.write(f"WARNING: No JSON found in response\n\n")
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            with self.file_lock:
                with open(self.output_file, 'a') as f:
                    f.write(f"ERROR: {e}\n\n")
            
        logger.warning("Returning empty entities and relationships")
        return [], []
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Azure OpenAI"""
        logger.debug(f"Getting embedding for text of length {len(text)}")
        try:
            # Apply rate limiting before making the request
            rate_limiter.wait_if_needed()
            
            response = client.embeddings.create(
                model=embedding_deployment,
                input=text
            )
            logger.debug(f"Successfully got embedding with {len(response.data[0].embedding)} dimensions")
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            logger.warning("Using random embedding as fallback")
            # Return random embedding as fallback
            return list(np.random.rand(1536))
    
    def build_graph_from_csv(self, csv_path: str):
        """Build knowledge graph from CSV file"""
        logger.info("Converting CSV to text...")
        text = self.csv_to_text(csv_path)
        
        logger.info("Chunking text...")
        self.chunks = self.chunk_text(text)
        
        logger.info(f"Processing {len(self.chunks)} chunks with 50 concurrent workers...")
        all_entities = []
        all_relationships = []
        
        # Process chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            # Submit all chunks for processing
            chunk_futures = {
                executor.submit(self.process_chunk, (i, chunk)): i 
                for i, chunk in enumerate(self.chunks)
            }
            
            # Collect results as they complete
            completed_chunks = {}
            for future in concurrent.futures.as_completed(chunk_futures):
                chunk_index, entities, relationships = future.result()
                completed_chunks[chunk_index] = (entities, relationships)
                
                logger.info(f"Completed chunk {chunk_index + 1}/{len(self.chunks)}")
        
        # Process results in order to maintain consistency
        logger.info("Processing completed chunks in order...")
        for i in range(len(self.chunks)):
            entities, relationships = completed_chunks[i]
            all_entities.extend(entities)
            all_relationships.extend(relationships)
            
            # Log progress to file
            with open(self.output_file, 'a') as f:
                f.write(f"CHUNK {i+1}/{len(self.chunks)} PROCESSED - Total entities so far: {len(all_entities)}, Total relationships so far: {len(all_relationships)}\n\n")
            
            logger.info(f"Processed chunk {i+1}/{len(self.chunks)} - Running totals: {len(all_entities)} entities, {len(all_relationships)} relationships")
        
        # Build graph
        logger.info("Building graph structure...")
        logger.info(f"Adding {len(all_entities)} entities to graph...")
        
        # Log final summary to file
        with open(self.output_file, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"FINAL EXTRACTION SUMMARY - {datetime.now()}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Total entities extracted: {len(all_entities)}\n")
            f.write(f"Total relationships extracted: {len(all_relationships)}\n")
            f.write(f"Processing {len(self.chunks)} chunks complete.\n")
            f.write(f"{'='*60}\n\n")
        
        entities_added = 0
        for entity in all_entities:
            entity_id = entity['name']
            self.graph.add_node(entity_id, **entity)
            self.entity_map[entity_id] = entity
            entities_added += 1
            
            if entities_added % 100 == 0:
                logger.info(f"Added {entities_added}/{len(all_entities)} entities to graph")
            
            # Get embedding for entity
            entity_text = f"{entity['name']} ({entity['type']}): {entity.get('description', '')}"
            logger.debug(f"Getting embedding for entity: {entity_id}")
            self.embeddings[entity_id] = self.get_embedding(entity_text)
        
        logger.info(f"Adding {len(all_relationships)} relationships to graph...")
        edges_added = 0
        for rel in all_relationships:
            if rel['source'] in self.graph and rel['target'] in self.graph:
                self.graph.add_edge(rel['source'], rel['target'], **rel)
                self.relationship_map[rel['source']].append(rel)
                self.relationship_map[rel['target']].append(rel)
                edges_added += 1
            else:
                logger.debug(f"Skipping relationship - missing nodes: {rel['source']} -> {rel['target']}")
        
        logger.info(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        logger.info(f"Successfully added {edges_added} out of {len(all_relationships)} relationships")
        
        # Log final graph stats to file
        with open(self.output_file, 'a') as f:
            f.write(f"FINAL GRAPH STATISTICS - {datetime.now()}\n")
            f.write(f"Nodes in graph: {self.graph.number_of_nodes()}\n")
            f.write(f"Edges in graph: {self.graph.number_of_edges()}\n")
            f.write(f"Entities processed: {entities_added}\n")
            f.write(f"Relationships added: {edges_added}/{len(all_relationships)}\n")
            f.write(f"{'='*60}\n\n")
    
    def create_community_summaries(self):
        """Create summaries for communities in the graph"""
        logger.info("Creating community summaries...")
        
        # Detect communities using Louvain method
        try:
            import community as community_louvain
            communities = community_louvain.best_partition(self.graph)
            logger.info("Using Louvain method for community detection")
        except ImportError:
            logger.warning("Community package not available, using connected components fallback")
            # Fallback to simple connected components
            communities = {}
            for i, component in enumerate(nx.connected_components(self.graph)):
                for node in component:
                    communities[node] = i
        
        # Group nodes by community
        community_groups = defaultdict(list)
        for node, comm_id in communities.items():
            community_groups[comm_id].append(node)
        
        logger.info(f"Found {len(community_groups)} communities")
        
        # Create summaries for each community
        self.community_summaries = {}
        communities_processed = 0
        
        for comm_id, nodes in community_groups.items():
            if len(nodes) > 1:
                logger.info(f"Creating summary for community {comm_id} with {len(nodes)} nodes")
                
                # Get entities in this community
                entities_desc = []
                for node in nodes[:10]:  # Limit to first 10 nodes
                    entity = self.entity_map.get(node, {})
                    entities_desc.append(f"{entity.get('name', node)} ({entity.get('type', 'Unknown')})")
                
                summary_prompt = f"""Create a brief summary of this group of related entities:
                {', '.join(entities_desc)}
                
                Provide a 1-2 sentence summary of what connects these entities."""
                
                try:
                    logger.debug(f"Requesting community summary for community {comm_id}")
                    
                    # Apply rate limiting before making the request
                    rate_limiter.wait_if_needed()
                    
                    response = client.chat.completions.create(
                        model=deployment,
                        messages=[
                            {"role": "system", "content": "You are an expert at summarizing relationships between entities."},
                            {"role": "user", "content": summary_prompt}
                        ],
                        max_completion_tokens=90000,
                    )
                    
                    summary_text = response.choices[0].message.content
                    
                    # Log community summary to file
                    with open(self.output_file, 'a') as f:
                        f.write(f"\n{'='*40}\n")
                        f.write(f"COMMUNITY {comm_id} SUMMARY - {datetime.now()}\n")
                        f.write(f"{'='*40}\n")
                        f.write(f"Nodes in community: {len(nodes)}\n")
                        f.write(f"Entities: {', '.join(entities_desc)}\n")
                        f.write(f"Summary: {summary_text}\n")
                        f.write(f"{'='*40}\n\n")
                    
                    self.community_summaries[comm_id] = {
                        "summary": summary_text,
                        "nodes": nodes,
                        "embedding": self.get_embedding(summary_text)
                    }
                    
                    communities_processed += 1
                    logger.info(f"Community {comm_id} summary created ({communities_processed} total)")
                    
                except Exception as e:
                    logger.error(f"Error creating community summary for community {comm_id}: {e}")
                    with open(self.output_file, 'a') as f:
                        f.write(f"ERROR creating summary for community {comm_id}: {e}\n\n")
        
        logger.info(f"Created {communities_processed} community summaries")
    
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
        
        logger.info(f"Graph saved to {output_path}")
        
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

    def process_chunk(self, chunk_data: Tuple[int, str]) -> Tuple[int, List[Dict], List[Dict]]:
        """Process a single chunk and return its index with extracted entities and relationships"""
        chunk_index, text_chunk = chunk_data
        logger.info(f"Processing chunk {chunk_index + 1}...")
        
        entities, relationships = self.extract_entities_relationships(text_chunk)
        
        # Log progress to file with thread safety
        with self.file_lock:
            with open(self.output_file, 'a') as f:
                f.write(f"CHUNK {chunk_index + 1} COMPLETE - Entities: {len(entities)}, Relationships: {len(relationships)}\n")
        
        logger.info(f"Chunk {chunk_index + 1} complete - Extracted {len(entities)} entities, {len(relationships)} relationships")
        return chunk_index, entities, relationships

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
        logger.error(f"CSV file not found: {csv_path}")
        logger.info("Please provide a valid CSV file path.")

if __name__ == "__main__":
    main()