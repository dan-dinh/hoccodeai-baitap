#################################################
# Author:   Dan Dinh                            #
# Date:     2025-03-27                          #
# Exercise: 6 with Stable Diffusion with API    #
# Description: Main file for the FastAPI app    #
#################################################

import base64
import io
import services

from fastapi.responses import StreamingResponse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import ImageRequest

# Create the FastAPI app
app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    """
    Root endpoint for the API
    Parameters:
    - None
    Returns:
    - dict: A dictionary containing a message
    """
    return {"message": "Welcome to Stable Diffusion API"}

@app.post("/api/v1/generate/")
async def generate_image(img_request: ImageRequest):
    """
    API endpoint to generate an image based on user inputs
    Parameters:
    - img_request (ImageRequest): An ImageRequest object containing user inputs
    Returns:
    - StreamingResponse: A StreamingResponse object containing the generated image
    """
    # Generate image based on user inputs
    image = await services.generate_image(img_request=img_request)

    # Convert image to memory stream to send back to client
    memory_stream = io.BytesIO()
    image.save(memory_stream, format="PNG")
    memory_stream.seek(0)

    # Return an image with PNG extesion (media type = 'image/png') from API
    return StreamingResponse(memory_stream, media_type="image/png")

@app.post('/api/v1/generatebase64')
async def generate_base64_image(img_request: ImageRequest):
    """
    API endpoint to generate image based on user inputs
    Parameters:
    - img_request (ImageRequest): An ImageRequest object containing user inputs
    Returns:
    - dict: A dictionary containing the generated image
    """
    # Generate image based on user inputs
    image = await services.generate_image(img_request=img_request)

    # Convert to base64 to send back to client
    buffered = io.BytesIO()
    image.save(buffered, format='JPEG')
    img_str = base64.b64encode(buffered.getvalue())

    # Return an image string under base64 encoded
    return {"message": img_str}