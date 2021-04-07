

#Append the directory to your python path using sys:
import torch
import os
import yaml
from tqdm import tqdm
import cv2
from glob import glob
import os
import pandas as pd
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from dataset.data_getter import __init__config, TruckDataset
import os.path as osp
import os
def get_lat_lon(path):  
  
  #get image file name (tail)
  head, tail = os.path.split(path)
  #split the path name into a pair root and ext:
  lat_lon = os.path.splitext(tail)[0]
  lat = lat_lon.split("_")[0]
  lon = lat_lon.split("_")[1]
  lat_lon = []
  lat_lon.append(lat)
  lat_lon.append(lon)
  return lat_lon

def predict(config_file):
    if not osp.exists(config_file["model"]["output_csv"]):
        res = pd.DataFrame(columns=["numb_truck","lat","lon"])
    else:
        res = pd.read_csv(config_file["model"]["output_csv"])
    cfg = get_cfg()
    cfg.merge_from_file("models/output/config.yml")
    cfg.MODEL.WEIGHTS = 'models/output/model_final.pth' # Set path model .pth
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config_file["model"]["score_thresh_test"]
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = 'gpu'
    else:
        cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.OUTPUT_CSV = config_file["model"]["output_csv"] 
    predictor = DefaultPredictor(cfg)
    numb_truck = []
    lats= []
    lons = []
    thresh =torch.tensor([0.99])
    imgs_pred = glob(osp.join(config_file["data_getter"]["output_img_dir"],"*.png"))
    for d in tqdm(imgs_pred):   
        im = cv2.imread(d)
        outputs = predictor(im)
        #% confident in prediction (scores)
        scores = outputs["instances"].scores
        #count number of prediction that has condifent >= threshold
        num_instances = torch.sum((scores >= thresh).int())
        num_instances = num_instances.cpu().data.numpy() #convert to numpy to get the value
        numb_truck.append(num_instances)

        #find lat/lon of image
        lat_lon= []
        lat_lon = get_lat_lon(d)
        lats.append(lat_lon[0])
        lons.append(lat_lon[1])
        # result.append((lat_lon[0], lat_lon[1], num_instances))


    sub_df = pd.DataFrame(columns=["numb_truck","lat","lon"])
    sub_df['numb_truck'] = numb_truck
    sub_df['lat'] = lats
    sub_df['lon'] = lons
    #sub_df = pd.concat([res, sub_df])
    res = pd.concat([res,sub_df])
    print(res)
    res.to_csv(config_file["model"]["output_csv"] , index=False,float_format="%.15f")

config_file = __init__config()
if __name__ == "__main__":
    truck_dataset = TruckDataset(config_file)
    print(truck_dataset)
    while truck_dataset != "Success":
        truck_dataset = TruckDataset(config_file)
        predict(config_file)
        files = glob(config_file["data_getter"]["output_img_dir"]+"*")
        print("Deleting Files")
        for f in files:
            os.remove(f)
    #print(truck_dataset)
    predict(config_file)
    