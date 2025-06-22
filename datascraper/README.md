# AI Local Search Agent

An intelligent local search agent that uses OpenAI as a bridge to access Yelp Fusion API functionality without requiring a Yelp API key.

## Features

- 🔍 **Natural Language Search**: Search for businesses using conversational queries
- 🤖 **AI-Powered**: Uses OpenAI GPT-4 to understand and process search requests
- 🌐 **Web Interface**: Beautiful, responsive web UI for easy interaction
- 📱 **REST API**: Programmatic access to search functionality
- 🎯 **Smart Results**: AI-generated realistic business data with proper formatting
- 💡 **No Yelp API Key Required**: Uses OpenAI to simulate Yelp API responses

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up OpenAI API Key

```bash
export OPENAI_API_KEY='your-openai-api-key-here'
```

### 3. Run the Application

```bash
cd datascraper
python main.py
```

### 4. Access the Web Interface

Open your browser and go to: http://localhost:5000

## Usage Examples

### Web Interface
- Search for "pizza restaurants in San Francisco"
- Find "coffee shops near me"
- Look for "high-rated Italian restaurants"

### REST API
```bash
# Search for restaurants
curl "http://localhost:5000/api/search?term=restaurant&location=San%20Francisco&limit=5"

# Search for coffee shops
curl "http://localhost:5000/api/search?term=coffee&location=New%20York&limit=10"
```

### Programmatic Usage
```python
from datascraper.main import LocalSearchAgent
import os

# Initialize the agent
agent = LocalSearchAgent(os.getenv('OPENAI_API_KEY'))

# Search for businesses
result = agent.process_search_query("Find me the best pizza places in Chicago")
print(result['response'])
```

## Architecture

### Components

1. **YelpFusionBridge**: Simulates Yelp API calls using OpenAI
2. **LocalSearchAgent**: Main AI agent that processes natural language queries
3. **Flask Web App**: Provides web interface and REST API endpoints

### How It Works

1. **Query Processing**: User submits a natural language query
2. **AI Parsing**: OpenAI extracts search parameters (term, location, preferences)
3. **Search Simulation**: AI generates realistic business results mimicking Yelp API
4. **Response Generation**: AI creates natural language summaries of results
5. **Presentation**: Results displayed in beautiful web interface

## API Endpoints

### POST /search
Process natural language search queries

**Request:**
```json
{
  "query": "Find pizza restaurants in San Francisco"
}
```

**Response:**
```json
{
  "query": "Find pizza restaurants in San Francisco",
  "search_params": {
    "term": "pizza",
    "location": "San Francisco",
    "limit": 10
  },
  "results": {
    "businesses": [...],
    "total": 3
  },
  "response": "I found 3 great pizza places in San Francisco..."
}
```

### GET /api/search
Direct search with parameters

**Parameters:**
- `term`: Search term (optional)
- `location`: Location to search (default: San Francisco)
- `limit`: Number of results (default: 10)

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Customization
You can modify the business templates in `_generate_realistic_results()` to include different types of businesses or customize the response format.

## Features in Detail

### Natural Language Understanding
The agent can understand queries like:
- "Show me Italian restaurants"
- "Find coffee shops near downtown"
- "What are the best rated pizza places?"
- "Restaurants open now in New York"

### Realistic Data Generation
- Proper business names and categories
- Realistic ratings and review counts
- Accurate address formatting
- Distance calculations
- Price indicators

### Smart Response Generation
- Contextual responses based on search results
- Highlights top recommendations
- Provides helpful suggestions
- Natural conversational tone

## Development

### Project Structure
```
datascraper/
├── main.py          # Main application file
├── requirements.txt # Python dependencies
└── README.md       # This file
```

### Adding New Features
1. Extend the `YelpFusionBridge` class for new API endpoints
2. Add new search parameters in `search_businesses()`
3. Update the web interface for new functionality
4. Add corresponding API endpoints

## Troubleshooting

### Common Issues

1. **"Agent not initialized"**: Make sure `OPENAI_API_KEY` is set
2. **Import errors**: Install dependencies with `pip install -r requirements.txt`
3. **Port already in use**: Change the port in `main.py` or kill existing processes

### Debug Mode
The application runs in debug mode by default. Check the console for detailed logs.

## License

This project is open source and available under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions, please open an issue on the repository. 