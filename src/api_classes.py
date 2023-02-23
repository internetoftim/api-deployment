# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from pydantic import BaseModel

# Summarization
class SummarizationInput(BaseModel):
    documents: str


# Question/Answer
class QA(BaseModel):
    context: str
    question: str

class StableDiffusion2Request(StableDiffusionRequest):
    negative_prompt: str = None
    guidance_scale: float = 9
    num_inference_steps: int = 25
