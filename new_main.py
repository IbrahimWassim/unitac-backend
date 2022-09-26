__author__ = "Wassim Brahim"
__copyright__ = "Copyright 2022, UNITAC"
__email__ = "wassim.brahim@un.org"
__status__ = "Production"

from email import header
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from os.path import join,exists
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

from log_management import log

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
    input_names = glob.glob(join(folder_path, "*.tif"))
    input_names.extend(glob.glob(join(folder_path, "*.tiff")))
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


def create_tiles(image_path):
    """
    cuts the big image (from the path) into tiles with tile size as stated in the global variables. the images are
    squared.
    the tiles will be saved under /images_tiles/image_name/...png
    @todo verify if the tiling works with triangles images.
    """
    log.info(f"Start creating tiles for the image {image_path} ...")
    tilling_start_time = datetime.now()
    global tile_dir, loaded_model
    global tile_size
    global image_shape_x, image_shape_y
    if not exists(tile_dir):
        os.makedirs(tile_dir)
    img = np.array(PILImage.create(image_path))
    image_shape_x, image_shape_y, _ = img.shape
    img_name = image_path.split("\\")[-1]
    if exists(join(tile_dir, img_name)):
        filelist = glob.glob(join(tile_dir, img_name, img_name + "*.png"))
        for f in filelist:
            os.remove(f)
    else:
        os.makedirs(join(tile_dir, img_name))
    # Cut tiles and save them
    for i in range(image_shape_x // tile_size):
        for j in range(image_shape_y // tile_size):
            img_tile = img[
                       i * tile_size: (i + 1) * tile_size, j * tile_size: (j + 1) * tile_size
                       ]
            Image.fromarray(img_tile).save(
                f"{join(tile_dir, img_name)}/{img_name}_000{i * (image_shape_x // tile_size) + j}.png"
            )
    log.info(f"Created tiles for image: {image_path}")
    tilling_end_time = datetime.now()
    log.info(f"Tiling time: {(tilling_end_time - tilling_start_time)}")

