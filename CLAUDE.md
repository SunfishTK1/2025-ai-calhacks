# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a digital clone project creating a comprehensive graph representation of the entire world, starting with Berkeley. The system combines vector embeddings with GraphRAG to enable natural language search and immersive user placement within the graph structure.

## Tech Stack

- **Backend**: FastAPI server
- **AI/ML Platform**: Azure AI Foundry Portal
- **Frontend**: TypeScript with React
- **AI Agents**: Python Claude fetch AI agentic workflow
- **LLM**: Gemini
- **Graph Processing**: NetworkX (current prototype)
- **Embeddings**: Vector embeddings for semantic search
- **Visualization**: Plotly (interactive), Matplotlib (static)

## Current Architecture

The repository currently contains a prototype Visual Graph RAG implementation:

### Core Components

- **VisualGraphRAG class**: Foundation for the knowledge graph system that will scale to represent Berkeley
- **Entity extraction**: Currently regex-based, will be enhanced with AI-powered entity recognition
- **Graph structure**: NetworkX undirected graph foundation for the world representation
- **Search capabilities**: Natural language interface for graph exploration
- **User positioning**: Framework for placing users within the graph context

### Key Methods

- `add_document()`: Processes geographic/location data and builds entity relationships
- `visualize_matplotlib()`: Static visualization of the world graph with user highlighting
- `visualize_plotly_interactive()`: Interactive exploration of the Berkeley digital twin
- `visualize_retrieval_process()`: Shows how natural language queries retrieve relevant graph sections
- `create_metrics_dashboard()`: Analytics for the digital world representation

## Development Roadmap

1. **Phase 1**: Enhance entity extraction with Azure AI services
2. **Phase 2**: Integrate vector embeddings for semantic similarity
3. **Phase 3**: Build FastAPI backend with Gemini integration
4. **Phase 4**: Develop React frontend for immersive graph navigation
5. **Phase 5**: Implement Claude AI agents for intelligent world querying

## Running the Current Prototype

```bash
python grag.py
```

This demonstrates the foundational graph visualization and search capabilities that will power the Berkeley digital clone.

## Future Integration Points

- FastAPI endpoints for graph queries and user positioning
- Azure AI Foundry for advanced entity recognition and embeddings
- React components for immersive world navigation
- Claude AI agents for intelligent assistance within the digital world
- Gemini integration for natural language understanding