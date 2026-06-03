import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Load the hidden API key from the .env file
load_dotenv()

# 2. Initialize the client (It automatically looks for GOOGLE_API_KEY in the environment)
client = genai.Client()

# 3. Test the connection with a simple prompt
def test_connection():
    print("Testing connection to Gemini...")
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents='Hello! Are you ready to start building agents?',
    )
    print(f"Response: {response.text}")

if __name__ == "__main__":
    test_connection()