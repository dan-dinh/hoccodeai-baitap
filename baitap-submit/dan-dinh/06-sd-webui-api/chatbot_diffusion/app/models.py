#################################################
# Author:   Dan Dinh                            #
# Date:     2025-03-27                          #
# Exercise: 6 with Stable Diffusion with API    #
# Description: Models for the FastAPI app       #
#################################################

from typing import Optional
from pydantic import BaseModel

class ImageRequest(BaseModel):
    """
    Class for image request
    """
    prompt: str
    num_inference_steps: Optional[int] = 25
    guidance_scale: Optional[float] = 7.5
    width: Optional[int] = 512
    height: Optional[int] = 512
    negative_prompt: Optional[str] = """
            watermark, text, censored, deformed, bad anatomy, disfigured, poorly drawn face, mutated, extra limb, ugly, 
            poorly drawn hands, missing limb, floating limbs, disconnected limbs, disconnected head, malformed hands, 
            long neck, mutated hands and fingers, missing fingers, cropped, worst quality, low quality, mutation, huge calf, 
            fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, 
            fused fingers, abnormal eye proportion, abnormal hands, abnormal legs, abnormal feet, abnormal fingers, 
            painting, extra fingers, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, 
            missing lips, ugly face, distorted face, extra legs
        """

class APIResponse(BaseModel):
    """
    Class for API response
    """
    image: str # base64 for image