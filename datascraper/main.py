import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template_string, request
from openai import AzureOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Business:
    id: str
    name: str
    rating: float
    price: str
    phone: str
    display_phone: str
    distance: float
    url: str
    image_url: str
    location: Dict[str, Any]
    categories: List[Dict[str, str]]
    coordinates: Dict[str, float]
    address: str
    city: str
    state: str
    zip_code: str
    country: str
    review_count: int


class YelpFusionBridge:
    """Bridge class that simulates Yelp Fusion API calls through OpenAI"""

    def __init__(self, openai_api_key: str):
        print("openai key", openai_api_key)
        self.client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint="https://2025-ai-hackberkeley.openai.azure.com/",
            api_version="2024-12-01-preview",  # Match this with the latest SDK-supported version,
            azure_deployment="o4-mini",
        )

    def search_businesses(
        self,
        term: Optional[str] = None,
        location: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        radius: int = 40000,
        categories: Optional[str] = None,
        price: Optional[str] = None,
        open_now: Optional[bool] = None,
        sort_by: str = "best_match",
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        Search for businesses using Yelp Fusion API simulation
        """
        try:
            # Create a function definition for Yelp API search
            function_definition = {
                "name": "search_businesses",
                "description": "Search for businesses on Yelp using various criteria",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "term": {
                            "type": "string",
                            "description": "Search term for businesses",
                        },
                        "location": {
                            "type": "string",
                            "description": "Location to search in (city, state, etc.)",
                        },
                        "latitude": {
                            "type": "number",
                            "description": "Latitude coordinate",
                        },
                        "longitude": {
                            "type": "number",
                            "description": "Longitude coordinate",
                        },
                        "radius": {
                            "type": "integer",
                            "description": "Search radius in meters (max 40000)",
                        },
                        "categories": {
                            "type": "string",
                            "description": "Comma-separated list of category aliases",
                        },
                        "price": {
                            "type": "string",
                            "description": "Price range: 1, 2, 3, or 4",
                        },
                        "open_now": {
                            "type": "boolean",
                            "description": "Whether to only return businesses that are open now",
                        },
                        "sort_by": {
                            "type": "string",
                            "enum": [
                                "best_match",
                                "rating",
                                "review_count",
                                "distance",
                            ],
                            "description": "Sort method",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of results to return (max 50)",
                        },
                    },
                    "required": [],
                },
            }

            # Build the search query
            search_params = {}
            if term:
                search_params["term"] = term
            if location:
                search_params["location"] = location
            if latitude is not None:
                search_params["latitude"] = latitude
            if longitude is not None:
                search_params["longitude"] = longitude
            if radius:
                search_params["radius"] = radius
            if categories:
                search_params["categories"] = categories
            if price:
                search_params["price"] = price
            if open_now is not None:
                search_params["open_now"] = open_now
            if sort_by:
                search_params["sort_by"] = sort_by
            if limit:
                search_params["limit"] = limit

            # Create the prompt for OpenAI
            prompt = f"""
            You are a Yelp Fusion API expert. Based on the search parameters provided, 
            generate realistic business search results that would be returned by the Yelp API.
            
            Search Parameters: {json.dumps(search_params, indent=2)}
            
            Please generate a JSON response that mimics the Yelp Fusion API search endpoint.
            Include realistic business data with proper formatting, ratings, reviews, and location information.
            The response should follow the exact structure of Yelp's API response.
            
            Generate {limit} business results with varied ratings, prices, and categories.
            """

            # Call OpenAI to generate the response
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Yelp Fusion API expert. Generate realistic business search results in JSON format.",
                    },
                    {"role": "user", "content": prompt},
                ],
                functions=[function_definition],
                function_call={"name": "search_businesses"},
            )

            # Extract the function call response
            function_call = response.choices[0].message.function_call
            if function_call and function_call.name == "search_businesses":
                # Parse the arguments and generate realistic results
                args = json.loads(function_call.arguments)
                return self._generate_realistic_results(args)

            return {"error": "Failed to generate search results"}

        except Exception as e:
            logger.error(f"Error in search_businesses: {str(e)}")
            return {"error": f"Search failed: {str(e)}"}

    def _generate_realistic_results(
        self, search_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate realistic business search results"""

        # Sample business data templates
        business_templates = [
            {
                "id": "restaurant-1",
                "name": "The Golden Fork",
                "rating": 4.5,
                "price": "$$",
                "phone": "+1-555-0123",
                "display_phone": "(555) 012-3456",
                "distance": 1200.5,
                "url": "https://www.yelp.com/biz/the-golden-fork",
                "image_url": "https://s3-media2.fl.yelpcdn.com/bphoto/sample1.jpg",
                "location": {
                    "address1": "123 Main St",
                    "address2": "",
                    "address3": "",
                    "city": "San Francisco",
                    "state": "CA",
                    "zip_code": "94102",
                    "country": "US",
                    "display_address": [
                        "123 Main St",
                        "San Francisco, CA 94102",
                    ],
                },
                "categories": [
                    {"alias": "restaurants", "title": "Restaurants"},
                    {"alias": "italian", "title": "Italian"},
                ],
                "coordinates": {"latitude": 37.7749, "longitude": -122.4194},
                "review_count": 156,
            },
            {
                "id": "cafe-1",
                "name": "Brew & Bean",
                "rating": 4.2,
                "price": "$",
                "phone": "+1-555-0456",
                "display_phone": "(555) 045-6789",
                "distance": 800.2,
                "url": "https://www.yelp.com/biz/brew-bean",
                "image_url": "https://s3-media2.fl.yelpcdn.com/bphoto/sample2.jpg",
                "location": {
                    "address1": "456 Coffee Ave",
                    "address2": "",
                    "address3": "",
                    "city": "San Francisco",
                    "state": "CA",
                    "zip_code": "94103",
                    "country": "US",
                    "display_address": [
                        "456 Coffee Ave",
                        "San Francisco, CA 94103",
                    ],
                },
                "categories": [
                    {"alias": "coffee", "title": "Coffee & Tea"},
                    {
                        "alias": "breakfast_brunch",
                        "title": "Breakfast & Brunch",
                    },
                ],
                "coordinates": {"latitude": 37.7849, "longitude": -122.4094},
                "review_count": 89,
            },
            {
                "id": "pizza-1",
                "name": "Slice of Heaven",
                "rating": 4.7,
                "price": "$$",
                "phone": "+1-555-0789",
                "display_phone": "(555) 078-9012",
                "distance": 1500.8,
                "url": "https://www.yelp.com/biz/slice-of-heaven",
                "image_url": "https://s3-media2.fl.yelpcdn.com/bphoto/sample3.jpg",
                "location": {
                    "address1": "789 Pizza Blvd",
                    "address2": "",
                    "address3": "",
                    "city": "San Francisco",
                    "state": "CA",
                    "zip_code": "94104",
                    "country": "US",
                    "display_address": [
                        "789 Pizza Blvd",
                        "San Francisco, CA 94104",
                    ],
                },
                "categories": [
                    {"alias": "pizza", "title": "Pizza"},
                    {"alias": "italian", "title": "Italian"},
                ],
                "coordinates": {"latitude": 37.7949, "longitude": -122.3994},
                "review_count": 234,
            },
        ]

        # Generate results based on search parameters
        term = search_params.get("term", "").lower()
        location = search_params.get("location", "San Francisco")
        limit = min(search_params.get("limit", 20), 50)

        # Filter and customize results based on search term
        results = []
        for i, template in enumerate(business_templates):
            if i >= limit:
                break

            # Customize based on search term
            business = template.copy()
            if term:
                if "pizza" in term:
                    business["name"] = f"Pizza Place {i + 1}"
                    business["categories"] = [
                        {"alias": "pizza", "title": "Pizza"}
                    ]
                elif "coffee" in term or "cafe" in term:
                    business["name"] = f"Coffee Shop {i + 1}"
                    business["categories"] = [
                        {"alias": "coffee", "title": "Coffee & Tea"}
                    ]
                elif "restaurant" in term:
                    business["name"] = f"Restaurant {i + 1}"
                    business["categories"] = [
                        {"alias": "restaurants", "title": "Restaurants"}
                    ]
                else:
                    business["name"] = f"{term.title()} Place {i + 1}"

            # Customize location
            if location and location != "San Francisco":
                business["location"]["city"] = location
                business["location"]["display_address"] = [
                    business["location"]["address1"],
                    f"{location}, CA 94102",
                ]

            # Vary ratings and distances
            business["rating"] = round(
                3.5 + (i * 0.3) + (hash(business["name"]) % 10) * 0.1, 1
            )
            business["distance"] = round(
                500 + (i * 200) + (hash(business["name"]) % 100), 1
            )
            business["review_count"] = (
                50 + (i * 20) + (hash(business["name"]) % 100)
            )

            results.append(business)

        return {
            "businesses": results,
            "total": len(results),
            "region": {
                "center": {"latitude": 37.7749, "longitude": -122.4194}
            },
        }

    def get_business_details(self, business_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific business"""
        try:
            # Generate detailed business information
            business_details = {
                "id": business_id,
                "name": f"Business {business_id}",
                "image_url": "https://s3-media2.fl.yelpcdn.com/bphoto/sample.jpg",
                "is_claimed": True,
                "is_closed": False,
                "url": f"https://www.yelp.com/biz/{business_id}",
                "phone": "+1-555-0000",
                "display_phone": "(555) 000-0000",
                "review_count": 150,
                "categories": [
                    {"alias": "restaurants", "title": "Restaurants"},
                    {"alias": "italian", "title": "Italian"},
                ],
                "rating": 4.5,
                "location": {
                    "address1": "123 Business St",
                    "address2": "",
                    "address3": "",
                    "city": "San Francisco",
                    "state": "CA",
                    "zip_code": "94102",
                    "country": "US",
                    "display_address": [
                        "123 Business St",
                        "San Francisco, CA 94102",
                    ],
                },
                "coordinates": {"latitude": 37.7749, "longitude": -122.4194},
                "photos": [
                    "https://s3-media2.fl.yelpcdn.com/bphoto/sample1.jpg",
                    "https://s3-media2.fl.yelpcdn.com/bphoto/sample2.jpg",
                ],
                "price": "$$",
                "hours": [
                    {
                        "open": [
                            {
                                "is_overnight": False,
                                "start": "1100",
                                "end": "2200",
                                "day": 0,
                            },
                            {
                                "is_overnight": False,
                                "start": "1100",
                                "end": "2200",
                                "day": 1,
                            },
                            {
                                "is_overnight": False,
                                "start": "1100",
                                "end": "2200",
                                "day": 2,
                            },
                            {
                                "is_overnight": False,
                                "start": "1100",
                                "end": "2200",
                                "day": 3,
                            },
                            {
                                "is_overnight": False,
                                "start": "1100",
                                "end": "2300",
                                "day": 4,
                            },
                            {
                                "is_overnight": False,
                                "start": "1100",
                                "end": "2300",
                                "day": 5,
                            },
                            {
                                "is_overnight": False,
                                "start": "1200",
                                "end": "2100",
                                "day": 6,
                            },
                        ],
                        "hours_type": "REGULAR",
                        "is_open_now": True,
                    }
                ],
                "transactions": ["delivery", "pickup"],
                "special_hours": [],
            }

            return business_details

        except Exception as e:
            logger.error(f"Error in get_business_details: {str(e)}")
            return {"error": f"Failed to get business details: {str(e)}"}


class LocalSearchAgent:
    """AI Agent for Local Search using Yelp Fusion Bridge"""

    def __init__(self, openai_api_key: str):
        self.yelp_bridge = YelpFusionBridge(openai_api_key)
        self.client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint="https://2025-ai-hackberkeley.openai.azure.com/",
            api_version="2024-12-01-preview",  # Match this with the latest SDK-supported version
            azure_deployment="o4-mini",
        )

    def process_search_query(self, user_query: str) -> Dict[str, Any]:
        """Process a natural language search query and return results"""
        try:
            # Use OpenAI to parse the user query and extract search parameters
            system_prompt = """
            You are a local search assistant that helps users find businesses. 
            Parse the user's query and extract relevant search parameters for Yelp API.
            
            Extract ONLY the following information:
            - Search term (what they're looking for)
            - Location (where to search)
            
            Return the parameters in JSON format.
            """

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
            )

            # Parse the response to extract search parameters
            try:
                # Try to extract JSON from the response
                content = response.choices[0].message.content
                if "{" in content and "}" in content:
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    params = json.loads(content[start:end])
                else:
                    # Fallback: extract parameters manually
                    params = self._extract_params_from_text(
                        content, user_query
                    )
            except:
                params = self._extract_params_from_text(
                    response.choices[0].message.content, user_query
                )

            # Perform the search
            print("here are the params I extracted...", params)

            search_results = self.yelp_bridge.search_businesses(**params)

            # Generate a natural language response
            response_text = self._generate_response(
                user_query, search_results, params
            )

            return {
                "query": user_query,
                "search_params": params,
                "results": search_results,
                "response": response_text,
            }

        except Exception as e:
            logger.error(f"Error processing search query: {str(e)}")
            return {
                "error": f"Failed to process search query: {str(e)}",
                "query": user_query,
            }

    def _extract_params_from_text(
        self, text: str, original_query: str
    ) -> Dict[str, Any]:
        """Extract search parameters from text response"""
        params = {}

        # Simple keyword extraction
        query_lower = original_query.lower()

        # Extract location
        location_keywords = ["in", "near", "at", "around"]
        for keyword in location_keywords:
            if keyword in query_lower:
                parts = query_lower.split(keyword)
                if len(parts) > 1:
                    params["location"] = parts[1].strip()
                    break

        # Extract search term
        if "restaurant" in query_lower:
            params["term"] = "restaurant"
        elif "coffee" in query_lower or "cafe" in query_lower:
            params["term"] = "coffee"
        elif "pizza" in query_lower:
            params["term"] = "pizza"
        else:
            # Extract the main search term
            words = original_query.split()
            if len(words) > 0:
                params["term"] = words[0]

        # Set defaults
        if "location" not in params:
            params["location"] = "San Francisco"
        if "limit" not in params:
            params["limit"] = 10

        return params

    def _generate_response(
        self, query: str, results: Dict[str, Any], params: Dict[str, Any]
    ) -> str:
        """Generate a natural language response based on search results"""
        try:
            if "error" in results:
                return f"I'm sorry, I couldn't find any results for '{query}'. Please try a different search term or location."

            businesses = results.get("businesses", [])
            if not businesses:
                return f"I couldn't find any {params.get('term', 'businesses')} in {params.get('location', 'the specified area')}."

            # Generate response using OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful local search assistant. Provide a natural, conversational response about the search results.",
                    },
                    {
                        "role": "user",
                        "content": f"""
                    Generate a helpful response for the user's query: "{query}"
                    
                    Search parameters: {json.dumps(params)}
                    Found {len(businesses)} businesses:
                    {json.dumps(businesses, indent=2)}
                    
                    Provide a natural response that summarizes the results and highlights the top recommendations.
                    """,
                    },
                ],
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I found {len(businesses)} results for your search. Here are the top recommendations."


# Flask web application
app = Flask(__name__)

# Initialize the agent
agent = None


@app.route("/")
def home():
    """Home page with search interface"""
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Local Search Agent</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            h1 {
                text-align: center;
                color: #333;
                margin-bottom: 30px;
                font-size: 2.5em;
            }
            .search-form {
                display: flex;
                gap: 15px;
                margin-bottom: 30px;
                flex-wrap: wrap;
            }
            .search-input {
                flex: 1;
                padding: 15px;
                border: 2px solid #ddd;
                border-radius: 10px;
                font-size: 16px;
                min-width: 300px;
            }
            .search-button {
                padding: 15px 30px;
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                cursor: pointer;
                transition: transform 0.2s;
            }
            .search-button:hover {
                transform: translateY(-2px);
            }
            .results {
                margin-top: 30px;
            }
            .business-card {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                border-left: 5px solid #667eea;
                transition: transform 0.2s;
            }
            .business-card:hover {
                transform: translateX(5px);
            }
            .business-name {
                font-size: 1.5em;
                font-weight: bold;
                color: #333;
                margin-bottom: 10px;
            }
            .business-info {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 15px;
            }
            .info-item {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .rating {
                color: #f39c12;
                font-weight: bold;
            }
            .price {
                color: #27ae60;
                font-weight: bold;
            }
            .address {
                color: #7f8c8d;
            }
            .loading {
                text-align: center;
                padding: 20px;
                color: #666;
            }
            .error {
                background: #e74c3c;
                color: white;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .ai-response {
                background: #e8f4fd;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                border-left: 5px solid #3498db;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 AI Local Search Agent</h1>
            <p style="text-align: center; color: #666; margin-bottom: 30px;">
                Powered by OpenAI + Yelp Fusion Bridge
            </p>
            
            <form class="search-form" id="searchForm">
                <input type="text" 
                       class="search-input" 
                       id="searchQuery" 
                       placeholder="Search for restaurants, coffee shops, pizza places, etc. in any city..."
                       required>
                <button type="submit" class="search-button">🔍 Search</button>
            </form>
            
            <div id="results" class="results"></div>
        </div>

        <script>
            document.getElementById('searchForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const query = document.getElementById('searchQuery').value;
                const resultsDiv = document.getElementById('results');
                
                // Show loading
                resultsDiv.innerHTML = '<div class="loading">🔍 Searching for the best local businesses...</div>';
                
                try {
                    const response = await fetch('/search', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ query: query })
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        resultsDiv.innerHTML = `<div class="error">❌ ${data.error}</div>`;
                        return;
                    }
                    
                    // Display AI response
                    let html = `<div class="ai-response">🤖 <strong>AI Assistant:</strong> ${data.response}</div>`;
                    
                    // Display business results
                    if (data.results && data.results.businesses) {
                        data.results.businesses.forEach(business => {
                            html += `
                                <div class="business-card">
                                    <div class="business-name">${business.name}</div>
                                    <div class="business-info">
                                        <div class="info-item">
                                            <span class="rating">⭐ ${business.rating}/5</span>
                                            <span>(${business.review_count} reviews)</span>
                                        </div>
                                        <div class="info-item">
                                            <span class="price">💰 ${business.price}</span>
                                        </div>
                                        <div class="info-item">
                                            <span>📞 ${business.display_phone}</span>
                                        </div>
                                        <div class="info-item">
                                            <span>📍 ${business.distance}m away</span>
                                        </div>
                                    </div>
                                    <div class="address">
                                        ${business.location.display_address.join(', ')}
                                    </div>
                                    <div style="margin-top: 10px;">
                                        ${business.categories.map(cat => `<span style="background: #e1f5fe; padding: 4px 8px; border-radius: 15px; margin-right: 5px; font-size: 12px;">${cat.title}</span>`).join('')}
                                    </div>
                                </div>
                            `;
                        });
                    }
                    
                    resultsDiv.innerHTML = html;
                    
                } catch (error) {
                    resultsDiv.innerHTML = `<div class="error">❌ Error: ${error.message}</div>`;
                }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)


@app.route("/search", methods=["POST"])
def search():
    """API endpoint for search requests"""
    try:
        data = request.get_json()
        query = data.get("query", "")

        if not query:
            return jsonify({"error": "Query is required"}), 400

        if not agent:
            return jsonify(
                {
                    "error": "Agent not initialized. Please set OPENAI_API_KEY environment variable."
                }
            ), 500

        result = agent.process_search_query(query)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        return jsonify({"error": f"Search failed: {str(e)}"}), 500


@app.route("/api/search", methods=["GET"])
def api_search():
    """REST API endpoint for programmatic access"""
    try:
        term = request.args.get("term")
        location = request.args.get("location", "San Francisco")
        limit = int(request.args.get("limit", 10))

        if not agent:
            return jsonify({"error": "Agent not initialized"}), 500

        results = agent.yelp_bridge.search_businesses(
            term=term, location=location, limit=limit
        )

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error in API search: {str(e)}")
        return jsonify({"error": f"API search failed: {str(e)}"}), 500


def main():
    """Main function to run the application"""
    global agent

    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY environment variable is required")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        return

    # Initialize the agent
    agent = LocalSearchAgent(openai_api_key)
    print("✅ AI Local Search Agent initialized successfully!")
    print("🌐 Starting web server...")
    print("📱 Open your browser and go to: http://localhost:8080")

    # Run the Flask app
    app.run(debug=True, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
