# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from PIL import Image
from config import settings
from diffusers import DPMSolverMultistepScheduler
import sys
import random
import logging

logger = logging.getLogger()

is_stub = settings.is_stub()
logger.info(f"Hello, StableDiffusion2Text2Img (STUBBED = {is_stub})")


class SD2Text2ImgModelPipeline:
    def __init__(self):
        if is_stub:
            self._pipe = None
        else:
            import torch
            from models.sd_model_wrapper import IPUStableDiffusionPipeline

            self._pipe = IPUStableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                revision="fp16",
                torch_dtype=torch.float16,
                ipu_config={
                    "matmul_proportion": [0.06, 0.1, 0.1, 0.1],
                    "executable_cache_dir": "./exe_cache/stablediffusion2-text2img",
                },
                requires_safety_checker=False,
            )
            self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self._pipe.scheduler.config
            )
            self._pipe.enable_attention_slicing()

    def __call__(self, inputs):
        if is_stub:
            images = [
                Image.new(
                    "RGB",
                    (768, 768),
                    random.choice(["blue", "red", "green", "orange", "black", "pink"]),
                )
            ]
            return {"result": images}
        else:
            import torch

            ret = self._pipe(
                prompt=inputs["prompt"],
                guidance_scale=inputs["guidance_scale"],
                num_inference_steps=inputs["num_inference_steps"],
                generator=torch.manual_seed(inputs["random_seed"]),
            )
            return {"result": ret["images"]}


pipe = SD2Text2ImgModelPipeline()


def compile(pipe):
    if not is_stub:
        pipe(
            {
                "prompt": "Big red dog",
                "random_seed": 31337,
                "guidance_scale": 9,
                "num_inference_steps": 25,
            }
        )
        return
