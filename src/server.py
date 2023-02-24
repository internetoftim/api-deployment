# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from fastapi import (
    FastAPI,
    Response,
    HTTPException,
    File,
    UploadFile,
    Security,
    Depends,
    Request,
)

from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from api_classes import *
import time
from ipu_worker import IPUWorkerGroup
from config import settings

app = FastAPI()
models = settings.server_models
w = IPUWorkerGroup(model_list=models)


@app.on_event("startup")
def startup_event():
    print("Running the following models:", models)
    w.start()
    return


@app.on_event("shutdown")
def shutdown_event():
    w.stop()
    return


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
async def home():
    return {"message": "Health Check Passed!"}


@app.post("/summarization", include_in_schema="summarization" in models)
def run_summarization(model_input: SummarizationInput):
    latency = time.time()

    data_dict = model_input.dict()
    w.workers["summarization"].feed(data_dict)
    result = w.workers["summarization"].get_result()

    latency = time.time() - latency

    headers = {"metrics-latency": str(latency)}
    response = {
        "results": result["summary"],
    }
    return JSONResponse(content=response, headers=headers)


@app.post("/qa", include_in_schema="question_answering" in models)
def run_qa(model_input: QA):
    latency = time.time()

    data_dict = model_input.dict()
    w.workers["question_answering"].feed(data_dict)
    result = w.workers["question_answering"].get_result()

    latency = time.time() - latency

    headers = {"metrics-latency": str(latency)}
    response = {
        "results": result["answer"],
        "score": result["score"],
    }
    return JSONResponse(content=response, headers=headers)


@app.post(
    "/stable_diffusion_2_text2img",
    include_in_schema="stable_diffusion_2_text2img" in models,
)
def run_stable_diffusion_2_text2img(
    model_input: StableDiffusion2Request 
):
    start = time.time()

    data_dict = model_input.dict()
    w.workers["stable_diffusion_2_text2img"].feed(data_dict)
    result = w.workers["stable_diffusion_2_text2img"].get_result()

    latency = time.time() - start

    headers = {
        "metrics-worker-latency": str(latency),
        # Add model_latency to SD model
        # "metrics-model-latency": str(results["model_latency"]),
    }

    return prepare_image_response(result, headers, model_input.return_json)