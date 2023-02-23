# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from pydantic import BaseSettings
from typing import List


class Settings(BaseSettings):
    server_models: List[str] = ["question_answering","stable_diffusion_2_text2img"]

    class Config:
        env_file = "../.env"


settings = Settings()
