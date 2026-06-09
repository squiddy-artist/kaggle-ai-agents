import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

try:
    # 1. Load the hidden API key from the .env file
    # This automatically finds your .env file and loads the variables inside it
    load_dotenv()
# ----------------------------------------
    # Fetch the key from the local environment
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    
    # Check if the key was actually found
    if not GOOGLE_API_KEY:
        raise ValueError("The GOOGLE_API_KEY variable was not found in the environment.")
        
    print("✅ Gemini API key setup complete.")
    
except Exception as e:
    print(
        f"🔑 Authentication Error: Please make sure you have created a '.env' file and added 'GOOGLE_API_KEY' to it. Details: {e}"
    )

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