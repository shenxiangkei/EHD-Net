import cv2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os
# Get image
path=r'/home/wsjc/Sxkai/new/EHD-Net/data/NEU-DET/2/'
output_dir=r'/home/wsjc/Sxkai/new/EHD-Net/data/NEU-DET/out3/'
cfg = get_cfg()
cfg.merge_from_file("/home/wsjc/Sxkai/new/EHD-Net/heatmap/NEU/ft_3_3/config.yaml")
cfg.MODEL.WEIGHTS = "/home/wsjc/Sxkai/new/EHD-Net/heatmap/NEU/ft_3_3/model_0011199.pth"
# cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)
for item in os.listdir(path):
    im = cv2.imread(os.path.join(path,item))

    outputs = predictor(im)
# huizhi Result!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    v = Visualizer(im[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs[0]['instances'].to('cpu'))
    img = v.get_image()[:, :, ::-1]
    cv2.imwrite(os.path.join(output_dir,item), img)

