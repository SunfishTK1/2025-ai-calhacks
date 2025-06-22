#!/usr/bin/env python3
"""
Setup script for AI Local Search Agent
"""

import os
import platform
import subprocess
import sys


def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")

    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported")
        print("Please use Python 3.8 or higher")
        return False

    print(
        f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible"
    )
    return True


def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")

    try:
        # Install from requirements.txt
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def check_openai_key():
    """Check if OpenAI API key is set"""
    print("\n🔑 Checking OpenAI API key...")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable is not set")
        print("\nTo set it up:")

        if platform.system() == "Windows":
            print("1. Open Command Prompt or PowerShell")
            print("2. Run: set OPENAI_API_KEY=your-api-key-here")
        else:
            print("1. Open terminal")
            print("2. Run: export OPENAI_API_KEY='your-api-key-here'")
            print("3. Or add to ~/.bashrc or ~/.zshrc for persistence")

        print(
            "\nYou can get an API key from: https://platform.openai.com/api-keys"
        )
        return False

    # Mask the key for security
    masked_key = (
        api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    )
    print(f"✅ OpenAI API key is set: {masked_key}")
    return True


def test_installation():
    """Test if the installation works"""
    print("\n🧪 Testing installation...")

    try:
        # Test importing the main module
        sys.path.append("datascraper")
        from main import LocalSearchAgent, YelpFusionBridge

        print("✅ Successfully imported LocalSearchAgent and YelpFusionBridge")

        # Test initialization (without making API calls)
        if os.getenv("OPENAI_API_KEY"):
            try:
                agent = LocalSearchAgent(os.getenv("OPENAI_API_KEY"))
                print("✅ Successfully initialized LocalSearchAgent")
            except Exception as e:
                print(f"⚠️  Warning: Could not initialize agent: {e}")
                print("This might be due to invalid API key or network issues")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed correctly")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def create_env_file():
    """Create a .env file template"""
    print("\n📝 Creating .env file template...")

    env_content = """# AI Local Search Agent Environment Variables
# Replace 'your-api-key-here' with your actual OpenAI API key
OPENAI_API_KEY=your-api-key-here

# Optional: Flask configuration
FLASK_ENV=development
FLASK_DEBUG=True
"""

    try:
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Created .env file template")
        print("📝 Edit .env file and add your OpenAI API key")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False


def show_next_steps():
    """Show next steps for the user"""
    print("\n🎯 Next Steps:")
    print("=" * 50)

    print("1. Set your OpenAI API key:")
    if platform.system() == "Windows":
        print("   set OPENAI_API_KEY=your-api-key-here")
    else:
        print("   export OPENAI_API_KEY='your-api-key-here'")

    print("\n2. Test the installation:")
    print("   python test_agent.py")

    print("\n3. Run the demo:")
    print("   python demo.py")

    print("\n4. Start the web server:")
    print("   cd datascraper")
    print("   python main.py")

    print("\n5. Open your browser and go to:")
    print("   http://localhost:5000")

    print("\n📚 For more information, see README.md")


def main():
    """Main setup function"""
    print("🚀 AI Local Search Agent - Setup")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        return

    # Install dependencies
    if not install_dependencies():
        return

    # Check OpenAI key
    check_openai_key()

    # Test installation
    if not test_installation():
        return

    # Create .env file
    create_env_file()

    # Show next steps
    show_next_steps()

    print("\n🎉 Setup completed!")


if __name__ == "__main__":
    main()
