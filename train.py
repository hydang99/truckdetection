from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os 
import yaml
import os.path as osp
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer
from setup import register_dataset
from dataset.data_getter import TruckDataset, __init__config
"""
Input: 
config_file: "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" (by default)
dataset: Setting the parameters for the dataset
model: Setting the parameters for the model
"""

def train(config_file):
    """
    Train method for detectron model
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file["model"]["model_path"]))
    cfg.DATASETS.TRAIN = (config_file["dataset"]["train_path"])
    cfg.DATALOADER.NUM_WORKERS = config_file["dataset"]["num_workers"]
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file["model"]["model_path"])
    cfg.SOLVER.IMS_PER_BATCH = config_file["model"]["ims_per_batch"]
    cfg.SOLVER.BASE_LR = config_file["model"]["base_lr"]
    cfg.MODEL.DEVICE = 'cpu'
    cfg.SOLVER.MAX_ITER = config_file["model"]["max_iter"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config_file["model"]["num_classes"]
    cfg.OUTPUT_DIR = config_file["model"]["output_dir"]
   #evaluator = COCOEvaluator(config_file["dataset"]["train_path"])
    if not osp.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR,exist_ok = True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume = False)
    train_loader = build_detection_train_loader(cfg,config_file["dataset"]["train_path"])
    trainer.train()
   # print(inference_on_dataset(trainer.model, train_loader, evaluator))
    with open(cfg.OUTPUT_DIR + "model_config.yml", "w") as f:
        f.write(cfg.dump())
    """
    Evaluate part for trained model
    """
    evaluator = COCOEvaluator(config_file["dataset"]["test_path"])
    val_loader = build_detection_test_loader(cfg, config_file["dataset"]["test_path"])
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume = True)
    print(inference_on_dataset(trainer.model, val_loader, evaluator))
config = __init__config()
if __name__ == "__main__":
    register_dataset()
    train(config_file)
    
