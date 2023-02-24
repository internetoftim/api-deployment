# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from pydantic import BaseModel
import time
from typing import Dict, List


# Summarization
class SummarizationInput(BaseModel):
    documents: str


# Question/Answer
class QA(BaseModel):
    context: str
    question: str

# Stable Diffusion
class StableDiffusionRequest(BaseModel):
    prompt: str = "Big red dog"
    random_seed: int = int(time.time())
    guidance_scale: float = 7.5
    return_json: bool = False    

class StableDiffusion2Request(StableDiffusionRequest):
    negative_prompt: str = None
    guidance_scale: float = 9
    num_inference_steps: int = 25
