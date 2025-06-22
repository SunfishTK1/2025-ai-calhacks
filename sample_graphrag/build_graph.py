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

# Multi-model configuration
CHAT_MODELS = {
    "gpt-4.1-mini": {"tpm": 250000, "rpm": 250},
    "o4-mini": {"tpm": 100000, "rpm": 600},
    "gpt-4.1": {"tpm": 250000, "rpm": 250},
    "o3-mini": {"tpm": 2500000, "rpm": 250},
    "gpt-4o": {"tpm": 250000, "rpm": 1500}
}

EMBEDDING_MODELS = {
    "text-embedding-3-large": {"tpm": 150000, "rpm": 900}
}

# Rate limiting setup
class ModelRateLimiter:
    def __init__(self, model_name: str, max_requests_per_minute: int, max_tokens_per_minute: int):
        self.model_name = model_name
        self.max_requests = max_requests_per_minute
        self.max_tokens = max_tokens_per_minute
        self.requests = deque()
        self.tokens = deque()  # Store (timestamp, token_count) tuples
        self.lock = threading.Lock()
    
    def wait_if_needed(self, estimated_tokens=1000):
        with self.lock:
            now = time.time()
            
            # Remove requests older than 1 minute
            while self.requests and now - self.requests[0] > 60:
                self.requests.popleft()
            
            # Remove token records older than 1 minute
            while self.tokens and now - self.tokens[0][0] > 60:
                self.tokens.popleft()
            
            # Calculate current token usage
            current_tokens = sum(token_count for _, token_count in self.tokens)
            
            # Check if we need to wait for requests or tokens
            requests_sleep = 0
            tokens_sleep = 0
            
            if len(self.requests) >= self.max_requests:
                requests_sleep = 60 - (now - self.requests[0]) + 0.1
                
            if current_tokens + estimated_tokens > self.max_tokens:
                # Find when enough tokens will expire to allow this request
                tokens_needed = current_tokens + estimated_tokens - self.max_tokens
                for timestamp, token_count in self.tokens:
                    tokens_needed -= token_count
                    if tokens_needed <= 0:
                        tokens_sleep = max(0, 60 - (now - timestamp)) + 0.1
                        break
            
            # Sleep for the longer of the two limits
            sleep_time = max(requests_sleep, tokens_sleep)
            if sleep_time > 0:
                if requests_sleep > tokens_sleep:
                    logger.info(f"{self.model_name}: Request rate limit reached, sleeping for {sleep_time:.2f} seconds")
                else:
                    logger.info(f"{self.model_name}: Token rate limit reached ({current_tokens} tokens), sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                
                # Clean up again after sleeping
                now = time.time()
                while self.requests and now - self.requests[0] > 60:
                    self.requests.popleft()
                while self.tokens and now - self.tokens[0][0] > 60:
                    self.tokens.popleft()
            
            # Record this request
            self.requests.append(now)
    
    def record_tokens(self, token_count):
        """Record actual token usage after API call"""
        with self.lock:
            now = time.time()
            self.tokens.append((now, token_count))
            
            # Clean up old records
            while self.tokens and now - self.tokens[0][0] > 60:
                self.tokens.popleft()

class MultiModelManager:
    def __init__(self):
        # Create rate limiters for each model
        self.chat_limiters = {}
        for model_name, limits in CHAT_MODELS.items():
            self.chat_limiters[model_name] = ModelRateLimiter(
                model_name, limits["rpm"], limits["tpm"]
            )
        
        self.embedding_limiters = {}
        for model_name, limits in EMBEDDING_MODELS.items():
            self.embedding_limiters[model_name] = ModelRateLimiter(
                model_name, limits["rpm"], limits["tpm"]
            )
        
        # Calculate weighted distribution based on model capacities
        self._calculate_model_weights()
        
        # Model selection state
        self.model_counter = 0
        self.model_lock = threading.Lock()
    
    def _calculate_model_weights(self):
        """Calculate weighted distribution based on RPM and TPM limits"""
        # Use the minimum of RPM and TPM capacity (normalized) as the weight
        total_capacity = 0
        model_capacities = {}
        
        for model_name, limits in CHAT_MODELS.items():
            # Normalize capacity - assume average 2000 tokens per request (more realistic for entity extraction)
            rpm_capacity = limits["rpm"]
            tpm_capacity = limits["tpm"] // 2000  # Convert TPM to effective RPM
            
            # Use the limiting factor as the model's effective capacity
            effective_capacity = min(rpm_capacity, tpm_capacity)
            model_capacities[model_name] = effective_capacity
            total_capacity += effective_capacity
        
        # Create weighted distribution list
        self.weighted_models = []
        for model_name, capacity in model_capacities.items():
            # Add model multiple times based on its capacity ratio
            weight = max(1, round((capacity / total_capacity) * 100))
            self.weighted_models.extend([model_name] * weight)
        
        logger.info(f"Model weights calculated:")
        for model_name, capacity in model_capacities.items():
            percentage = (capacity / total_capacity) * 100
            logger.info(f"  {model_name}: {capacity} effective RPM ({percentage:.1f}% of workload)")
        
        logger.info(f"Total effective capacity: {total_capacity} RPM")
    
    def get_next_chat_model(self):
        """Get the next chat model using weighted distribution"""
        with self.model_lock:
            model = self.weighted_models[self.model_counter % len(self.weighted_models)]
            self.model_counter += 1
            return model
    
    def get_next_embedding_model(self):
        """Get the next available embedding model in round-robin fashion"""
        # Only one embedding model, so just return it
        return list(EMBEDDING_MODELS.keys())[0]
    
    def wait_for_chat_model(self, model_name: str, estimated_tokens: int):
        """Apply rate limiting for a specific chat model"""
        self.chat_limiters[model_name].wait_if_needed(estimated_tokens)
    
    def wait_for_embedding_model(self, model_name: str, estimated_tokens: int):
        """Apply rate limiting for a specific embedding model"""
        self.embedding_limiters[model_name].wait_if_needed(estimated_tokens)
    
    def record_chat_tokens(self, model_name: str, token_count: int):
        """Record token usage for a chat model"""
        self.chat_limiters[model_name].record_tokens(token_count)
    
    def record_embedding_tokens(self, model_name: str, token_count: int):
        """Record token usage for an embedding model"""
        self.embedding_limiters[model_name].record_tokens(token_count)

# Import itertools for cycling through models
import itertools

# Initialize multi-model manager
model_manager = MultiModelManager()

# Configuration
endpoint = os.getenv("ENDPOINT_URL", "https://2025-ai-hackberkeley.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4.1-mini")
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
        
        # Get the next available chat model
        model_name = model_manager.get_next_chat_model()
        
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
            logger.debug(f"Sending request to {model_name} for entity extraction")
            
            # Estimate token usage (rough approximation: 1 token ≈ 4 characters)
            estimated_tokens = (len(text_chunk) + len(prompt)) // 4 + 2000  # Add buffer for response
            
            # Apply rate limiting before making the request
            model_manager.wait_for_chat_model(model_name, estimated_tokens)
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured information from text."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=15000,
            )
            
            # Record actual token usage if available
            if hasattr(response, 'usage') and response.usage:
                actual_tokens = response.usage.total_tokens
                model_manager.record_chat_tokens(model_name, actual_tokens)
                logger.debug(f"Used {actual_tokens} tokens on {model_name}")
            else:
                # Fallback to estimated tokens
                model_manager.record_chat_tokens(model_name, estimated_tokens)
            
            result = response.choices[0].message.content
            logger.debug(f"Received response of {len(result)} characters from {model_name}")
            
            # Save raw model output to file
            with self.file_lock:
                with open(self.output_file, 'a') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"CHUNK EXTRACTION - {datetime.now()} - Model: {model_name}\n")
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
                
                logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships using {model_name}")
                return entities, relationships
            else:
                logger.warning(f"No JSON found in response from {model_name}")
                with self.file_lock:
                    with open(self.output_file, 'a') as f:
                        f.write(f"WARNING: No JSON found in response from {model_name}\n\n")
            
        except Exception as e:
            logger.error(f"Error extracting entities using {model_name}: {e}")
            with self.file_lock:
                with open(self.output_file, 'a') as f:
                    f.write(f"ERROR using {model_name}: {e}\n\n")
            
        logger.warning("Returning empty entities and relationships")
        return [], []
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Azure OpenAI"""
        logger.debug(f"Getting embedding for text of length {len(text)}")
        try:
            # Get the next available embedding model
            model_name = model_manager.get_next_embedding_model()
            
            # Estimate token usage for embeddings (typically much smaller than chat completions)
            estimated_tokens = len(text) // 4 + 10
            
            # Apply rate limiting before making the request
            model_manager.wait_for_embedding_model(model_name, estimated_tokens)
            
            response = client.embeddings.create(
                model=model_name,
                input=text
            )
            
            # Record token usage (embeddings usually return token count in usage)
            if hasattr(response, 'usage') and response.usage:
                actual_tokens = response.usage.total_tokens
                model_manager.record_embedding_tokens(model_name, actual_tokens)
                logger.debug(f"Used {actual_tokens} tokens for embedding with {model_name}")
            else:
                model_manager.record_embedding_tokens(model_name, estimated_tokens)
            
            logger.debug(f"Successfully got embedding with {len(response.data[0].embedding)} dimensions using {model_name}")
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
        
        # Calculate optimal concurrency based on weighted model distribution
        # Get the effective capacity from the model manager
        total_effective_rpm = sum(min(limits["rpm"], limits["tpm"] // 2000) for limits in CHAT_MODELS.values())
        
        # Conservative approach: Use 80% of total capacity to account for rate limiting overhead
        usable_capacity = int(total_effective_rpm * 0.8)
        
        # Each worker processes roughly 1 chunk per minute on average (including wait times)
        # So optimal workers ≈ effective RPM capacity
        optimal_workers = min(usable_capacity, 100)  # Cap at 100 for stability
        
        logger.info(f"Processing {len(self.chunks)} chunks with {optimal_workers} concurrent workers...")
        logger.info(f"Effective capacity: {total_effective_rpm} RPM, using {usable_capacity} RPM ({optimal_workers} workers)")
        logger.info(f"Weighted distribution will favor o3-mini (~37% of requests) due to superior token capacity")
        all_entities = []
        all_relationships = []
        
        # Process chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_workers) as executor:
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
        skipped_entities = 0
        valid_entities = []  # Store valid entities for parallel embedding processing
        
        for entity in all_entities:
            # Check if entity has required keys
            if not isinstance(entity, dict):
                logger.debug(f"Skipping non-dict entity: {entity}")
                skipped_entities += 1
                continue
                
            entity_name = entity.get('name')
            if not entity_name:
                logger.debug(f"Skipping entity with missing name: {entity}")
                skipped_entities += 1
                continue
                
            entity_id = entity_name
            self.graph.add_node(entity_id, **entity)
            self.entity_map[entity_id] = entity
            valid_entities.append((entity_id, entity))
            entities_added += 1
            
            if entities_added % 100 == 0:
                logger.info(f"Added {entities_added}/{len(all_entities)} entities to graph")
        
        if skipped_entities > 0:
            logger.warning(f"Skipped {skipped_entities} invalid entities")
        
        # Generate embeddings in parallel
        logger.info(f"Generating embeddings for {len(valid_entities)} entities using parallel processing...")
        self._generate_embeddings_parallel(valid_entities)
        
        logger.info(f"Adding {len(all_relationships)} relationships to graph...")
        edges_added = 0
        skipped_relationships = 0
        for rel in all_relationships:
            # Check if relationship has required keys
            if not isinstance(rel, dict):
                logger.debug(f"Skipping non-dict relationship: {rel}")
                skipped_relationships += 1
                continue
                
            source = rel.get('source')
            target = rel.get('target')
            
            if not source or not target:
                logger.debug(f"Skipping relationship with missing source/target: {rel}")
                skipped_relationships += 1
                continue
                
            if source in self.graph and target in self.graph:
                self.graph.add_edge(source, target, **rel)
                self.relationship_map[source].append(rel)
                self.relationship_map[target].append(rel)
                edges_added += 1
            else:
                logger.debug(f"Skipping relationship - missing nodes: {source} -> {target}")
                skipped_relationships += 1
        
        logger.info(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        logger.info(f"Successfully added {edges_added} out of {len(all_relationships)} relationships")
        if skipped_relationships > 0:
            logger.warning(f"Skipped {skipped_relationships} invalid relationships")
        
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
                    
                    # Get the next available chat model
                    model_name = model_manager.get_next_chat_model()
                    
                    # Estimate token usage for community summary
                    estimated_tokens = len(summary_prompt) // 4 + 500  # Smaller response expected
                    
                    # Apply rate limiting before making the request
                    model_manager.wait_for_chat_model(model_name, estimated_tokens)
                    
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are an expert at summarizing relationships between entities."},
                            {"role": "user", "content": summary_prompt}
                        ],
                        max_completion_tokens=15000,
                    )
                    
                    # Record actual token usage if available
                    if hasattr(response, 'usage') and response.usage:
                        actual_tokens = response.usage.total_tokens
                        model_manager.record_chat_tokens(model_name, actual_tokens)
                        logger.debug(f"Used {actual_tokens} tokens for community summary with {model_name}")
                    else:
                        model_manager.record_chat_tokens(model_name, estimated_tokens)
                    
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

    def _generate_embeddings_parallel(self, valid_entities: List[Tuple[str, Dict]]):
        """Generate embeddings for entities in parallel to maximize throughput"""
        
        # Calculate optimal workers for embeddings
        # text-embedding-3-large: 900 RPM, 150,000 TPM
        # Embeddings use ~10-50 tokens each, so TPM is rarely the limit
        embedding_rpm = 900
        embedding_workers = min(int(embedding_rpm * 0.8 / 60 * 10), 90, len(valid_entities))  # 10-second batches, cap at 90
        
        logger.info(f"Using {embedding_workers} workers for parallel embedding generation")
        
        def process_entity_embedding(entity_data):
            entity_id, entity = entity_data
            entity_text = f"{entity.get('name', '')} ({entity.get('type', '')}): {entity.get('description', '')}"
            embedding = self.get_embedding(entity_text)
            return entity_id, embedding
        
        # Process embeddings in parallel
        embeddings_completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=embedding_workers) as executor:
            # Submit all embedding tasks
            embedding_futures = {
                executor.submit(process_entity_embedding, entity_data): entity_data[0]
                for entity_data in valid_entities
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(embedding_futures):
                try:
                    entity_id, embedding = future.result()
                    self.embeddings[entity_id] = embedding
                    embeddings_completed += 1
                    
                    if embeddings_completed % 100 == 0:
                        logger.info(f"Generated {embeddings_completed}/{len(valid_entities)} embeddings")
                        
                except Exception as e:
                    entity_id = embedding_futures[future]
                    logger.error(f"Error generating embedding for {entity_id}: {e}")
                    # Use random embedding as fallback
                    self.embeddings[entity_id] = list(np.random.rand(1536))
        
        logger.info(f"Completed embedding generation for {embeddings_completed} entities")

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