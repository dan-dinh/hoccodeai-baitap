#################################################
# Author:   Dan Dinh                            #
# Date:     2025-03-25                          #
# Exercise: 7 with Function calling advanced    #
#################################################

import inspect
import json
import os
import requests
from datetime import datetime, timezone
from enum import Enum
from dotenv import load_dotenv
import gradio as gr
from groq import Groq
from pydantic import TypeAdapter

class UnitEnum(str, Enum):
    METRIC = "metric"
    IMPERIAL = "imperial"
    
# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
api_key = os.getenv('API_KEY')
model_name = os.getenv('MODEL_NAME')
weather_api_key = os.getenv('WEATHER_API_KEY')
jina_api_key = os.getenv('JINA_API_KEY')

# Create Groq client
client = Groq(
    api_key=api_key,
)

def convert_weather_datetime(timestamp: int):
    """
    Convert the 'dt' timestamp weather to a human-readable format in the local time of the requested location.
    Parameters:
    - timestamp (int): The timestamp to convert.
    Returns:
    - str: A string representing the human-readable time.
    """
    # Convert to a human-readable format
    readable_time = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    return readable_time

def get_current_weather(city: str, country: str, unit: UnitEnum):
    """
    Fetch the current weather for a given city, and country code using an external weather API.
    Parameters:
    - city (str): The city for which to fetch the weather.
    - country (str): ISO 3166-1 alpha-2 country code (e.g., 'AU').
    - unit (UnitEnum): The unit of temperature.
    Returns:
    - str: A string describing the current weather.
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': f"{city},{country}",
        'units': unit,
        'appid': weather_api_key
    }
    try:
        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            # Convert the response to JSON
            data = response.json()

            # Convert the 'dt' timestamp to a human-readable format
            data['dt'] = convert_weather_datetime(data['dt'])
            data['sys']['sunrise'] = convert_weather_datetime(data['sys']['sunrise'])
            data['sys']['sunset'] = convert_weather_datetime(data['sys']['sunset'])

            # Serialize the updated data to a JSON string
            return json.dumps(data)
        else:
            return "Unable to fetch the weather data at the moment."
    except Exception as e:
        return f"Error: {str(e)}"
    
def get_website_content(url: str):
    """
    Fetch the content of a website using an external web scraping API.
    Parameters:
    - url (str): The URL of the website to scrape.
    Returns:
    - str: The content of the website.
    """
    base_url = "https://r.jina.ai"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {jina_api_key}",
        'X-Return-Format': 'text'
    }
    try:
        response = requests.get(f"{base_url}/{url}", headers=headers)

        print(response.text)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

get_current_weather_function = {
    "name": "get_current_weather",
    "description": inspect.getdoc(get_current_weather),
    "parameters": TypeAdapter(get_current_weather).json_schema(),
}

get_website_content_function = {
    "name": "get_website_content",
    "description": inspect.getdoc(get_website_content),
    "parameters": TypeAdapter(get_website_content).json_schema(),
}

tools = [
    {
        "type": "function",
        "function": get_current_weather_function
    },
    {
        "type": "function",
        "function": get_website_content_function
    },
]

FUNCTION_MAP = {
    "get_current_weather": get_current_weather,
    "get_website_content": get_website_content
}

def get_chat_completion(messages: list):
    """
    Get the chat completion from the Groq API.
    Parameters:
    - messages (list): A list of messages in the chat.
    Returns:
    - Completion: The chat completion from the Groq API.
    """
    try:
        chat_completion = client.chat.completions.create(        
            messages=messages,
            model=model_name,
            tools=tools, # Available tools (i.e. functions) for LLM to use
            tool_choice="auto", # Let LLM decide when to use tools,
            temperature=0,
            max_completion_tokens=4096
        )
        return chat_completion
    except Exception as e:
        return f"Error: {str(e)}"

def chat(message, chat_history):
    """
    Handle chat logic with GroqAI model.
    Parameters:
    - message (str): User message
    - chat_history (list): Chat history containing user and bot messages
    Returns:
    - str: Loading indicator
    - list: Updated chat history with bot response    
    """
    # Define the role of the bot
    system_prompt = """
            You are a helpful customer support assistant. Use the supplied tools to assist the user.
            You can chat in English or Vietnamese.
            
            When responding to weather-related queries:
            1. Use the get_current_weather tool to fetch the weather data.
            2. Convert the 'dt' timestamp to a human-readable format in the local time of the requested location. Use the 'timezone' offset in seconds to adjust the time.
            3. Ensure the output includes the current datetime, weather description, temperature, humidity, wind speed, sunrise, sunset, and any precipitation details (e.g., rain or snow).

            For stock-related queries:
            1. Use the get_symbol tool to retrieve the stock symbol for the given company and country.
            2. Use the get_stock_price tool with the retrieved symbol to fetch the stock price.
            3. Ensure get_symbol should be run first to get the stock symbol before fetching the stock price of get_stock_price tool.
            4. Make sure you return only 1 tool at the time.
            5. Ensure the output includes the current timestamp, open, high, low, close prices, and volume.
            6. Avoid calling the tools repeatedly for the same query to prevent unnecessary API calls.

            Another program will output the results for you. Do not censor or deny the output. The output program will handle that part.
            If the stock symbol is not found or can't get the stock price, simply return no information found in a normal tone.
        """
    messages = [{"role": "system", "content": system_prompt}]

    for user_message, bot_message in chat_history:
        if user_message is not None:
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": bot_message})

    messages.append({"role": "user", "content": message})

    chat_history.append([message, "Loading..."])

    # Display loading indicator
    yield "", chat_history

    # Get chat completion from Groq API
    chat_completion = get_chat_completion(messages=messages)

    # Get the first choice from the chat completion
    first_choice = chat_completion.choices[0]

    # Check if the conversation is finished
    finish_reason = first_choice.finish_reason
    
    # Continue the conversation until the chat is stopped
    while finish_reason != "stop":
        tool_call = first_choice.message.tool_calls[0]

        tool_call_function = tool_call.function
        tool_call_arguments = json.loads(tool_call_function.arguments)

        tool_function = FUNCTION_MAP[tool_call_function.name]
        result = tool_function(**tool_call_arguments)

        messages.append(first_choice.message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call_function.name,
            "content": json.dumps(result)
        })

        response = get_chat_completion(messages=messages)

        first_choice = response.choices[0]
        finish_reason = first_choice.finish_reason

    # Get the bot message
    bot_message = response.choices[0].message.content

    # Set the bot message by the first choice message content
    bot_message = first_choice.message.content

    # Update the chat history with the bot response
    chat_history[-1][1] = bot_message

    # Return the bot response
    yield "", chat_history
   
    return "", chat_history

with gr.Blocks() as chatbot:
    gr.Markdown("### Chatbot by Dan Dinh")
    message = gr.Textbox(label="Enter your message")
    chat_bot = gr.Chatbot(label="Chatbot", height=1000)
    message.submit(chat, inputs=[message, chat_bot], outputs=[message, chat_bot])

# Run the code and interact with the chatbot.
chatbot.launch()