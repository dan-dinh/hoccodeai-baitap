#################################################
# Author:   Dan Dinh                            #
# Date:     2024-11-27                          #
# Exercise: 2 with Together AI                  #
# Question: 2 with array messages               #
#################################################

import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
api_key = os.getenv('API_KEY')

# Create Groq client
client = Groq(
    api_key=api_key,
)

print("Groq: What can I help you with?")

messages = []

# Chat loop, break when user input is empty
while True:
    question = input('You: ')

    if not question:
        break
    
    # Add user question to messages array
    messages.append({"role": "user", "content": question})

    # Create chat completion
    chat_completion = client.chat.completions.create(        
        messages=messages,
        model="llama3-8b-8192",
        stream=True,
    )

    return_message = ""
    # Print response from chat completion
    for chunk in chat_completion:
        response = chunk.choices[0].delta.content or ""
        print(response, end="")
        return_message += response

    # Add response to messages array
    messages.append({"role": "system", "content": return_message})
    print("\n")