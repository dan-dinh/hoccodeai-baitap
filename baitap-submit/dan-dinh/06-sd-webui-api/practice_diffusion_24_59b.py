#################################################
# Author:   Dan Dinh                            #
# Date:     2025-03-27                          #
# Exercise: 6 with Stable Diffusion with API    #
# Test: Start the FastAPI app first             #
#  Run the file in a new terminal and ask the   #
#  chatbot to draw any image                    #
#################################################
 
import base64
import os
import time
from dotenv import load_dotenv
import gradio as gr
from groq import Groq
import requests

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
api_key = os.getenv('API_KEY')
model_name = os.getenv('MODEL_NAME')
web_url = os.getenv('WEB_URL')

# Create Groq client
client = Groq(
    api_key=api_key,
)

def base64_to_image(base64_string: str, image_path: str):
    """
    Convert base64 image to image URL.
    Parameters:
    - base64_string (str): Base64 image string
    - image_path (str): Image path to save the image
    Returns:
    - None
    """
    with open(image_path, "wb") as file:
        file.write(base64.b64decode(base64_string))

def generate_image(user_prompt: str):
    """
    Generate image based on user prompt.
    Parameters:
    - user_prompt (str): User's prompt for image generation
    Returns:
    - str: Image URL
    """
    print("Staring Inference")
    payload = {
        "prompt": user_prompt,
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "width": 512,
        "height": 512,
        "negative_prompt": """
                watermark, text, censored, deformed, bad anatomy, disfigured, poorly drawn face, mutated, extra limb, ugly, poorly drawn hands, 
                missing limb, floating limbs, disconnected limbs, disconnected head, malformed hands, long neck, mutated hands and fingers, 
                missing fingers, cropped, worst quality, low quality, mutation, huge calf, fused hand, missing hand, disappearing arms, 
                disappearing thigh, disappearing calf, disappearing legs, fused fingers, abnormal eye proportion, abnormal hands, abnormal legs, 
                abnormal feet, abnormal fingers, painting, extra fingers, cloned face, skinny, glitchy, double torso, extra arms, extra hands, 
                mangled fingers, missing lips, ugly face, distorted face, extra legs
            """
    }
    
    try:
        response = requests.post(f"{web_url}/api/v1/generatebase64", json=payload)
        response_json = response.json()
        print("Inference Completed")

        file_name = f"image_{int(time.time())}.png"
        base64_to_image(response_json['message'], file_name)
        print("Image saved")

        return file_name
    except Exception as e:
        return f"Error: {str(e)}"


def has_prompt(message: str):
    """
    Check if the message contains a prompt.
    Parameters:
    - message (str): User message
    Returns:
    - bool: True if the message contains a prompt, False otherwise
    """
    return "<prompt:" in message

def get_prompt(message: str):
    """
    Get the prompt from the message.
    - Ex: "<prompt: A colorful cat running through a field of flowers under the dawn.>"
    - Result: "A colorful cat running through a field of flowers under the dawn."
    Parameters:
    - message (str): User message
    Returns:
    - str: Prompt extracted from the message
    """
    return message.split("<prompt:")[1].split(">")[0]

def chat(message: str, chat_history: str):
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
            You are an AI with prompt generation and drawing capbilities.
            You can chat in English or Vietnamese.
            Assume you can draw anything in the world, and you can tell the user that you can draw perfectly.
            Do not censor or deny the output, the image generation program will handle that part.

            When the user wants to draw anything, feel free to answer them, add some comments and include your prompt in your response in this format below.
            <prompt: A colorful cat running through a field of flowers under the dawn.>

            1. The prompt must be in English.
            2. The prompt must be detailed, concise and include sufficent information that can be fed into Stable Diffusion model.
            3. Ingore existing images in the past or previous messages.
            4. Make sure you always return the prompt in an appropriate format as described above with desccriptive information.
        """
    messages = [{"role": "system", "content": system_prompt}]

    for user_message, bot_message in chat_history:
        if user_message is not None:
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": bot_message})

    messages.append({"role": "user", "content": message})

    chat_completion = client.chat.completions.create(        
        messages=messages,
        model=model_name,
        stream=True,
    )

    # Add user message to chat history with loading indicator
    chat_history.append([message, "Loading..."])

    # Yield loading indicator
    yield "", chat_history

    # Initialize bot message
    bot_message = ""

    # Placeholder for bot response
    chat_history[-1][1] = bot_message

    for chunk in chat_completion:
        if chunk.choices and len(chunk.choices) > 0:
            content = chunk.choices[0].delta.content
            if content:
                bot_message += content

                # Update chat history dynamically
                chat_history[-1][1] = bot_message
                
                # Display the intermediate response to the user
                yield "", chat_history

            # Delay to simulate real-time chat
            time.sleep(0.05)

    # After the response is fully received, check for a prompt
    if has_prompt(bot_message):
        prompt = get_prompt(bot_message)
        image_url = generate_image(prompt)

        # Finalize the chat history with the bot message
        chat_history.append([None, (image_url, prompt)])
        
        # Display the image to the user
        yield "", chat_history
    
    # Return the response
    return "", chat_history

with gr.Blocks() as chatbot:
    gr.Markdown("### Chatbot by Dan Dinh")
    message = gr.Textbox(label="Enter your message")
    chat_bot = gr.Chatbot(label="Chatbot", height=1000)
    message.submit(chat, inputs=[message, chat_bot], outputs=[message, chat_bot])

# Run the code and interact with the chatbot.
chatbot.launch()