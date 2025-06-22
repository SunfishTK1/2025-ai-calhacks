#!/usr/bin/env python3
"""
Test script for the AI Local Search Agent
"""

import os
import sys


def test_agent():
    """Test the LocalSearchAgent functionality"""

    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is not set")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        return False

    try:
        # Import the agent
        sys.path.append("datascraper")
        from main import LocalSearchAgent

        print("✅ Successfully imported LocalSearchAgent")

        # Initialize the agent
        agent = LocalSearchAgent(os.getenv("OPENAI_API_KEY"))
        print("✅ Successfully initialized LocalSearchAgent")

        # Test a simple search
        print("\n🔍 Testing search functionality...")
        result = agent.process_search_query(
            "Find pizza restaurants in San Francisco"
        )

        if "error" in result:
            print(f"❌ Search failed: {result['error']}")
            return False

        print("✅ Search completed successfully!")
        print(
            f"📊 Found {len(result['results'].get('businesses', []))} businesses"
        )
        print(f"🤖 AI Response: {result['response'][:100]}...")

        # Test the Yelp bridge directly
        print("\n🔍 Testing Yelp bridge directly...")
        bridge_results = agent.yelp_bridge.search_businesses(
            term="coffee", location="New York", limit=3
        )

        if "error" in bridge_results:
            print(f"❌ Bridge test failed: {bridge_results['error']}")
            return False

        print("✅ Bridge test completed successfully!")
        print(
            f"📊 Found {len(bridge_results.get('businesses', []))} businesses"
        )

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(
            "Make sure you have installed the requirements: pip install -r requirements.txt"
        )
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_dependencies():
    """Test if all required dependencies are installed"""
    print("🔍 Testing dependencies...")

    required_packages = ["openai", "flask", "requests"]
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        return False

    print("✅ All dependencies are installed!")
    return True


def main():
    """Main test function"""
    print("🧪 AI Local Search Agent - Test Suite")
    print("=" * 50)

    # Test dependencies first
    if not test_dependencies():
        return

    print("\n" + "=" * 50)

    # Test the agent
    if test_agent():
        print(
            "\n🎉 All tests passed! The AI Local Search Agent is ready to use."
        )
        print("\nTo start the web server, run:")
        print("cd datascraper")
        print("python main.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
