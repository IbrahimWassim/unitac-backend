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

loadedModel = None
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
