import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import zipfile
import requests
import os
from detectron2.data.datasets import register_coco_instances
import random
from detectron2.data import DatasetCatalog, MetadataCatalog




#Get zip from link / get data from config["raw_link"]
def download_zip(url, save_path, chunk_size=128):
    r = requests.get(url, stream = True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size = chunk_size):
            fd.write(chunk)

#unzip to directory
def get_zip_dir(path_to_file, directory_to_extract_to):
    #Check if directory exists and make if not
    if not os.path.isdir(os.path.join(os.getcwd(), directory_to_extract_to)):
        os.mkdir(os.path.join(os.getcwd(), directory_to_extract_to))

    with zipFile.ZipFile(path_to_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)

#register data set with COCO format
def register_dataset():
    for d in ["train", "test"]:
            register_coco_instances(f"truck_{d}", {}, f"TruckDataset_TD/{d}.json", f"TruckDataset_TD/{d}") 

#show random sample
def show_random_sample(dataset):
    dataset_dicts = DatasetCatalog.get(dataset)
    truck_metadata = MetadataCatalog.get(dataset)

    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        v = Visualizer(img[:, :, ::-1], metadata=truck_metadata, scale=0.5)
        v = v.draw_dataset_dict(d)
        plt.figure(figsize = (14, 10))
        plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        plt.show()









