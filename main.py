import os
import cv2
import rasterio
import numpy as np
from rasterio.plot import show
from matplotlib import pyplot as plt
from PIL import Image

work_dir = os.path.expanduser("~")
data_dir = os.path.join(work_dir, "data", "dataset-cimat")
tiff_dir = os.path.join(data_dir, "tiff")
norm_dir = os.path.join(data_dir, "norm")
var_dir = os.path.join(data_dir, "var_jpg")
wind_dir = os.path.join(data_dir, "windfield_tiff")
mask_dir = os.path.join(data_dir, "binary_mask_gimp_png")

for fname in os.listdir(tiff_dir):
    print(fname)
