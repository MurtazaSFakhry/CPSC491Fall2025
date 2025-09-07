import os
from dotenv import load_dotenv

# Load variables from .env in project root
load_dotenv()

def get_api_key():
    """Fetch OpenAI API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment. Did you create a .env file?")
    return api_key
