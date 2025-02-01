#################################################
# Author:   Dan Dinh                            #
# Date:     2024-11-27                          #
# Exercise: 2 with Together AI                  #
# Question: 3 with article summary by url       #
#################################################

import os
from groq import Groq
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
api_key = os.getenv('API_KEY')

# Create Groq client
client = Groq(
    api_key=api_key,
)

# Get user input for URL
url = input("Enter the URL: ")

# Make a GET request to the URL
response = requests.get(url)

# Get the HTML content of the page
html = response.text

# Parse the HTML content
soup = BeautifulSoup(html, "html.parser")

# Extract the text from the main-detail div element
text = soup.find(id="main-detail").get_text()

# Define the prompt for the chat completion
prompt = f"\
            You are a journalist. Summarize the following article delimited by triple quotes, providing a brief summary of the article: \n \
            \"\"\"{text}\"\"\" \n \
            1. Make sure the summary is coherent, captures the essence of the article and less than 100 words. \n\
            2. Exclude any URLs or links. \n\
            3. Try to keep it as concise as possible. \n\
            4. Ignore any comments at the end of the article or unrelated text. \n\
            5. Translate it from English to Vietnamese and make sure it is easy to understand.\n\
            6. Do not write the English version.\n"

# Create chat completion
chat_completion = client.chat.completions.create(        
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    model="llama3-8b-8192",
    stream=True,
)

# Print response from chat completion
for chunk in chat_completion:
    response = chunk.choices[0].delta.content or ""
    print(response, end="")

print("\n")