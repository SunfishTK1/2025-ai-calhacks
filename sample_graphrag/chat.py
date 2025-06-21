import os
import base64
from openai import AzureOpenAI

from dotenv import load_dotenv
load_dotenv()

# Configuration
endpoint = os.getenv("ENDPOINT_URL", "https://2025-ai-hackberkeley.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "o4-mini")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

if not subscription_key:
    raise ValueError("Please set the AZURE_OPENAI_API_KEY environment variable")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2025-01-01-preview",  # Use a valid API version
)

def chat_with_azure_openai(message, conversation_history=None):
    """Send a message to Azure OpenAI and get a response"""
    if conversation_history is None:
        conversation_history = []
    
    # Add system message if it's the first interaction
    if not conversation_history:
        messages = [
            {"role": "system", "content": "You are an AI assistant that helps people find information."},
            {"role": "user", "content": message}
        ]
    else:
        messages = conversation_history + [{"role": "user", "content": message}]
    
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_completion_tokens=1000,  # Note: might need to use max_tokens instead of max_completion_tokens
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error calling Azure OpenAI: {e}")
        return None

def main():
    """Simple chat interface"""
    print("Azure OpenAI Chat Interface")
    print("Type 'quit' to exit")
    
    conversation_history = []
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'quit':
            break
        
        response = chat_with_azure_openai(user_input, conversation_history)
        
        if response:
            print(f"AI: {response}")
            conversation_history.extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response}
            ])
        else:
            print("Sorry, there was an error processing your request.")

if __name__ == "__main__":
    main()