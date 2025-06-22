#!/usr/bin/env python3
"""
Demo script for the AI Local Search Agent
Shows various usage examples and capabilities
"""

import os
import sys


def demo_basic_search():
    """Demonstrate basic search functionality"""
    print("🔍 Demo 1: Basic Search")
    print("-" * 40)

    try:
        sys.path.append("datascraper")
        from main import LocalSearchAgent

        agent = LocalSearchAgent(os.getenv("OPENAI_API_KEY"))

        # Example searches
        queries = [
            "Find pizza restaurants in San Francisco",
            "Show me coffee shops in New York",
            "What are the best Italian restaurants in Chicago?",
            "Find high-rated sushi places in Los Angeles",
        ]

        for query in queries:
            print(f"\nQuery: {query}")
            result = agent.process_search_query(query)

            if "error" in result:
                print(f"❌ Error: {result['error']}")
                continue

            print(
                f"✅ Found {len(result['results'].get('businesses', []))} businesses"
            )
            print(f"🤖 AI Response: {result['response'][:150]}...")

    except Exception as e:
        print(f"❌ Demo failed: {e}")


def demo_api_usage():
    """Demonstrate direct API usage"""
    print("\n🔍 Demo 2: Direct API Usage")
    print("-" * 40)

    try:
        sys.path.append("datascraper")
        from main import YelpFusionBridge

        bridge = YelpFusionBridge(os.getenv("OPENAI_API_KEY"))

        # Test different search parameters
        searches = [
            {"term": "restaurant", "location": "San Francisco", "limit": 5},
            {"term": "coffee", "location": "New York", "limit": 3},
            {"term": "pizza", "location": "Chicago", "limit": 2},
            {"categories": "italian", "location": "Los Angeles", "limit": 4},
        ]

        for i, search_params in enumerate(searches, 1):
            print(f"\nSearch {i}: {search_params}")
            results = bridge.search_businesses(**search_params)

            if "error" in results:
                print(f"❌ Error: {results['error']}")
                continue

            businesses = results.get("businesses", [])
            print(f"✅ Found {len(businesses)} businesses")

            for j, business in enumerate(businesses[:2], 1):  # Show first 2
                print(
                    f"  {j}. {business['name']} - ⭐ {business['rating']}/5 - {business['price']}"
                )
                print(f"     📍 {business['location']['display_address'][0]}")

    except Exception as e:
        print(f"❌ Demo failed: {e}")


def demo_business_details():
    """Demonstrate getting business details"""
    print("\n🔍 Demo 3: Business Details")
    print("-" * 40)

    try:
        sys.path.append("datascraper")
        from main import YelpFusionBridge

        bridge = YelpFusionBridge(os.getenv("OPENAI_API_KEY"))

        # Get details for a sample business
        business_id = "sample-restaurant-123"
        details = bridge.get_business_details(business_id)

        if "error" in details:
            print(f"❌ Error: {details['error']}")
            return

        print(f"✅ Business Details for {details['name']}:")
        print(f"   📞 Phone: {details['display_phone']}")
        print(
            f"   ⭐ Rating: {details['rating']}/5 ({details['review_count']} reviews)"
        )
        print(f"   💰 Price: {details['price']}")
        print(
            f"   📍 Address: {', '.join(details['location']['display_address'])}"
        )
        print(
            f"   🕒 Open Now: {'Yes' if details['hours'][0]['is_open_now'] else 'No'}"
        )

        # Show categories
        categories = [cat["title"] for cat in details["categories"]]
        print(f"   🏷️  Categories: {', '.join(categories)}")

    except Exception as e:
        print(f"❌ Demo failed: {e}")


def demo_natural_language():
    """Demonstrate natural language understanding"""
    print("\n🔍 Demo 4: Natural Language Understanding")
    print("-" * 40)

    try:
        sys.path.append("datascraper")
        from main import LocalSearchAgent

        agent = LocalSearchAgent(os.getenv("OPENAI_API_KEY"))

        # Complex natural language queries
        complex_queries = [
            "I'm looking for a romantic Italian restaurant with good wine selection in San Francisco",
            "Find me the cheapest coffee shops that are open late in New York",
            "What are the most popular pizza places with delivery in Chicago?",
            "Show me high-end sushi restaurants with outdoor seating in Los Angeles",
        ]

        for query in complex_queries:
            print(f"\nQuery: {query}")
            result = agent.process_search_query(query)

            if "error" in result:
                print(f"❌ Error: {result['error']}")
                continue

            print(f"✅ Parsed Parameters: {result['search_params']}")
            print(f"🤖 AI Response: {result['response'][:200]}...")

    except Exception as e:
        print(f"❌ Demo failed: {e}")


def demo_error_handling():
    """Demonstrate error handling"""
    print("\n🔍 Demo 5: Error Handling")
    print("-" * 40)

    try:
        sys.path.append("datascraper")
        from main import LocalSearchAgent

        agent = LocalSearchAgent(os.getenv("OPENAI_API_KEY"))

        # Test with invalid queries
        invalid_queries = [
            "",  # Empty query
            "   ",  # Whitespace only
            "Find restaurants in a very small town that probably doesn't exist",  # Unlikely location
        ]

        for query in invalid_queries:
            print(f"\nQuery: '{query}'")
            result = agent.process_search_query(query)

            if "error" in result:
                print(f"❌ Expected error: {result['error']}")
            else:
                print("✅ Query processed successfully")
                print(
                    f"   Found {len(result['results'].get('businesses', []))} businesses"
                )

    except Exception as e:
        print(f"❌ Demo failed: {e}")


def main():
    """Main demo function"""
    print("🎯 AI Local Search Agent - Demo Suite")
    print("=" * 60)

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is not set")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        return

    # Run all demos
    demos = [
        demo_basic_search,
        demo_api_usage,
        demo_business_details,
        demo_natural_language,
        demo_error_handling,
    ]

    for demo in demos:
        try:
            demo()
            print("\n" + "=" * 60)
        except KeyboardInterrupt:
            print("\n⏹️  Demo interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            print("=" * 60)

    print("\n🎉 Demo completed!")
    print("\nTo start the web server, run:")
    print("cd datascraper")
    print("python main.py")


if __name__ == "__main__":
    main()
