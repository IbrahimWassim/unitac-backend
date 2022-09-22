__author__ = "Wassim Brahim"
__copyright__ = "Copyright 2022, UNITAC"
__email__ = "wassim.brahim@un.org"
__status__ = "Production"

from email import header
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastai.vision.all import *
import pandas as pd
import numpy as np
import os
import glob
import shutil
from datetime import datetime
import pathlib
from natsort import os_sorted
import rasterio
import rasterio.features
import sys
import shapely.geometry
import geopandas as gpd

import log_management.log as log

loaded_model = None
input_names = []
output_folder = ""
tile_dir = "./image_tiles"
tile_size = 1000
predicted_dir = "./predicted_images"
image_shape_x = None
image_shape_y = None
origins = ["http://localhost:8080"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """
    start up event for the whole process to check hardware checkup are proper.
    """
    log.info("GPU available: " + str(torch.cuda.is_available()))
    log.info(torch.cuda.device(0))


@app.get("/ping")
def ping_pong():
    """
    Function to help the frontend note when the backend is ready. It is a health check.
    """
    headers = {"Access-Control-Allow-Origin": "*"}
    return JSONResponse(
        content={"started": True}, status_code=status.HTTP_200_OK, headers=headers
    )


@app.post("/exit")
def exit_process():
    """
    function to exit the process clean
    """
    os._exit(0)


@app.get("/uploadImages/")
def upload_images(folder_path: str):
    """
    function to get images from frontend in a loop from the folder_path.
    """
    global input_names
    log.info("Images will imported from: " + folder_path)
    input_names = glob.glob(os.path.join(folder_path, "*.tif"))
    input_names.extend(glob.glob(os.path.join(folder_path, "*.tiff")))
    headers = {"Access-Control-Allow-Origin": "*"}
    try:
        if len(input_names) > 0:
            log.info("Input Images are:")
            log.info(input_names[0])
            content = {"selectedImages": input_names}
            return JSONResponse(
                content=content, status_code=status.HTTP_200_OK, headers=headers
            )
        else:
            log.warn(f"No images found in {folder_path}")
            return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, headers=headers)
    except ValueError as err:
        log.warn(f"Unexpected {err=}, {type(err)=}")


@app.get("/loadModel/")
def load_model(model: str):
    """
    Loads the by default model, otherwise, loads the model is the paramter
    """
    global loaded_model
    # model = './models/exported-model.pkl' if not model else model
    try:
        loaded_model = load_learner(
            model,
            cpu=False if torch.cuda.is_available() else True,
            pickle_module=pickle,
        )
        log.info(f"Model {model} loaded successfully")
    except OSError as err:
        log.warn(f"Unexpected {err=}, {type(err)=}")
    content = {"selectedModel": model}
    headers = {"Access-Control-Allow-Origin": "*"}
    return JSONResponse(
        content=content, status_code=status.HTTP_200_OK, headers=headers
    )


@app.get("/loadOutputDir/")
def load_output(folder: str):
    """
    Loads the output folder on selection. The output folder is the one where we store the output shape files.
    """
    global output_folder
    log.info(f"The Shape files will be exported to: {output_folder}")
    output_folder = folder
    content = {"selectedOutput": output_folder}
    headers = {"Access-Control-Allow-Origin": "*"}
    return JSONResponse(
        content=content, status_code=status.HTTP_200_OK, headers=headers
    )

