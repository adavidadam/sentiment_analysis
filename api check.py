import openai
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import os

# Set the API key from environment

try:
    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, can you confirm the API key works?"}
    ])
    print("API Key is working!")
    print("Response:", response.choices[0].message.content)
except openai.AuthenticationError:
    print("API Key is invalid or not authorized.")
except Exception as e:
    print("An error occurred:", str(e))
