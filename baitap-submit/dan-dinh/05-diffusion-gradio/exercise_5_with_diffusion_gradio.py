#################################################
# Author:   Dan Dinh                            #
# Date:     2025-03-29                          #
# Exercise: 5 with Stable Diffusion             #
# Test: Run the app in the Terminal             #
#  enter prompt and select a model provider     #
#  and generate image image                     #
#################################################

import os
import random
from diffusers import DiffusionPipeline, StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from dotenv import load_dotenv
import torch
import gradio as gr

load_dotenv()

# Get API key and other environment variables
DIFFUSION_MODEL_PATH = os.getenv('DIFFUSION_MODEL_PATH')
MODEL_PROVIDERS = os.getenv('MODEL_PROVIDERS')
MODEL_HUGGING_FACE = os.getenv('MODEL_HUGGING_FACE')
MODEL_CIVIT_AI = os.getenv('MODEL_CIVIT_AI')

if not MODEL_PROVIDERS:
    raise ValueError("Can't find any model provider. Please define the model provider!")

if not MODEL_HUGGING_FACE and not MODEL_CIVIT_AI:
    raise ValueError("Can't find any model. Please define the model!")
    
# Split model providers
model_providers = MODEL_PROVIDERS.split(", ")
hugging_face_models = MODEL_HUGGING_FACE.split(", ")
civit_ai_models = MODEL_CIVIT_AI.split(", ")

# Get user inputs
negative_prompt = """
    watermark, text, censored, deformed, bad anatomy, disfigured, poorly drawn face, mutated, 
    extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, 
    disconnected head, malformed hands, long neck, mutated hands and fingers, missing fingers, 
    cropped, worst quality, low quality, mutation, huge calf, fused hand, missing hand, 
    disappearing arms, disappearing thigh, disappearing calf, disappearing legs, fused fingers, 
    abnormal eye proportion, abnormal hands, abnormal legs, abnormal feet, abnormal fingers, 
    painting, extra fingers, cloned face, skinny, glitchy, double torso, extra arms, extra hands, 
    mangled fingers, missing lips, ugly face, distorted face, extra legs
"""

def get_models(provider: str):
    """
    Get models based on provider
    Parameters:
    - provider (str): The name of the model provider
    Returns:
    - list: A list of models
    """
    if provider == "Hugging Face":
        return hugging_face_models
    elif provider == "Civit AI":
        return civit_ai_models
    else:
        return []

# Initialize with the first provider's models
default_provider = model_providers[0]
default_models = get_models(default_provider)

def is_file_exist(file_path: str):
    """
    Check file exist or not
    Parameters:
    - file_path (str): The path of the file
    Returns:
    - bool: True if the file exists
    """
    try:
        if os.path.exists(file_path):
            return True
        return False
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

def generate_image(user_prompt: str, negative_prompt: str, seed: int, steps: int, guidedance_scale: float, image_height: int, image_width: int):
    """
    Generate image based on user inputs
    Parameters:
    - user_prompt (str): User prompt
    - negative_prompt (str): Negative prompt
    - seed (int): Seed number
    - steps (int): Number of steps
    - guidedance_scale (float): Guidance scale
    - image_height (int): Image height
    - image_width (int): Image weight
    Returns:
    - gr.Image: Generated image
    - int: Seed number
    """
    global pipeline, device    

    # Check if the user provided a seed
    if seed is None or seed == -1:
        seed = random.randint(0, 2**32 - 1)  # Generate a random seed if not provided
        print(f"Using random seed: {seed}")
    else:
        print(f"Using user-defined seed: {seed}")

    try:
        # Set seed for reproducibility
        generator = torch.Generator(device=device).manual_seed(seed)

        # Generate image based on user inputs
        image = pipeline(prompt=user_prompt, 
                        num_inference_steps=steps, 
                        height=image_height, 
                        width=image_width, 
                        negative_prompt=negative_prompt,
                        generator=generator,
                        guidedance_scale=guidedance_scale).images[0]

        return image, seed
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def update_model_provider(selected_model_provider: str):
    """
    Update model provider
    Parameters:
    - selected_model_provider (str): The name of the selected model provider
    Returns:
    - str: A message regarding the updated model provider
    - gr.Dropdown: The updated model dropdown
    """
    if selected_model_provider == "Hugging Face":
        updated_models = MODEL_HUGGING_FACE.split(", ")
    elif selected_model_provider == "Civit AI":
        updated_models = MODEL_CIVIT_AI.split(", ")
    else:
        updated_models = []
    return f"Models updated to {selected_model_provider}", gr.Dropdown(choices=updated_models)


def update_model(selected_model_provider: str, selected_model: str):
    """
    Update model based on user selection
    Parameters:
    - selected_model_provider (str): The name of the selected model provider
    - selected_model (str): The name of the selected model
    Returns:
    - str: A message regarding the updated model
    """
    global pipeline, device

    # Define model path
    model_path = os.path.join(DIFFUSION_MODEL_PATH, f"{selected_model}.safetensors")

    try:
        if selected_model_provider == "Hugging Face":
            pipeline = DiffusionPipeline.from_pretrained(selected_model,
                                                        torch_dtype=torch.float16,
                                                        use_safetensors=True,
                                                        safety_checker=None,
                                                        requires_safety_checker=False)
        else:
            if not is_file_exist(model_path):
                return  f"Can't find the model with path: {model_path}"

            # Load Stable Diffusion model
            pipeline = StableDiffusionPipeline.from_single_file(model_path,
                                                        torch_dtype=torch.float16,
                                                        use_safetensors=True,
                                                        safety_checker=None,
                                                        requires_safety_checker=False)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load device to pipeline
        pipeline.to(device)

        # Load Euler Ancestral Discrete Scheduler
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

        return f"Model updated to {selected_model}"
    except Exception as e:
        return f"Error updating model: {str(e)}"

# Define Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Stable Diffusion with Gradio")
    with gr.Row():
        with gr.Column():
            user_prompt = gr.Textbox(lines=3, label="Prompt", placeholder="Enter your prompt: Example: a beautiful painting with a river and mountains")
            negative_prompt = gr.Textbox(lines=3, label="Negative Prompt", placeholder="Enter your negative prompt", value=negative_prompt)
            model_provider = gr.Dropdown(choices=model_providers, label="Model Provider", interactive=True)
            model = gr.Dropdown(choices=[], label="Model Name", interactive=True)
            seed = gr.Number(value=-1, label="Seed Number, Random seed = -1 (Note: using same seed number will generate same image)", precision=0)
            num_steps = gr.Number(value=20, label="Number of Inference Steps")
            guidedance_scale = gr.Number(value=7, label="Guidance Scale")
            image_height = gr.Number(value=768, label="Image Height (must be divisible by 8)")
            image_width = gr.Number(value=512, label="Image Width (must be divisible by 8)")
            generate_button = gr.Button(value="Generate Image")

        with gr.Column():
            output_message = gr.Textbox(value="Model not updated yet.", label="Selected model", interactive=True)
            output_image = gr.Image(label="Generated Image")
            output_seed = gr.Textbox(label="Seed Number")
    
    # Bind events
    model_provider.change(fn=update_model_provider, inputs=model_provider, outputs=[output_message, model])
    model.change(fn=update_model, inputs=[model_provider, model], outputs=output_message)
    generate_button.click(generate_image, inputs=[user_prompt, negative_prompt, seed, num_steps, guidedance_scale, image_height, image_width], outputs=[output_image, output_seed])

if __name__ == "__main__":
    demo.launch()