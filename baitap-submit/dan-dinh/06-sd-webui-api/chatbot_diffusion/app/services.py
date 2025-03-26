#################################################
# Author:   Dan Dinh                            #
# Date:     2025-03-27                          #
# Exercise: 6 with Stable Diffusion with API    #
# Description: Services for the FastAPI app     #
#################################################

from diffusers import DiffusionPipeline, StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from dotenv import load_dotenv
import torch
from PIL import Image
from models import ImageRequest

load_dotenv()

# Define pipleline and device
pipeline = DiffusionPipeline.from_pretrained("stablediffusionapi/anything-v5",
                                            torch_dtype=torch.float16,
                                            use_safetensors=True,
                                            safety_checker=None,
                                            requires_safety_checker=False,
                                            local_files_only=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load device to pipeline
pipeline.to(device)

# Load Euler Ancestral Discrete Scheduler
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

async def generate_image(img_request: ImageRequest) -> Image:
    """
    Generate image based on user inputs
    Parameters:
    - img_request (ImageRequest): An ImageRequest object containing user inputs
    Returns:
    - Image: Generated image
    """
    image: Image = pipeline(prompt=img_request.prompt, 
                    # strength=strength,
                    num_inference_steps=img_request.num_inference_steps,
                    height=img_request.height, 
                    width=img_request.width, 
                    negative_prompt=img_request.negative_prompt,
                    guidance_scale=img_request.guidance_scale).images[0]
    return image