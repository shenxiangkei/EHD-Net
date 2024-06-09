import os
import pickle
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import sys
from matplotlib import pyplot as plt
import torch
import tqdm


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)

    return objects


def calculate_confusion_matrix_values(confusion_matrix, class_index):
    TP = confusion_matrix[class_index, class_index]
    FN = np.sum(confusion_matrix[class_index, :]) - TP
    FP = np.sum(confusion_matrix[:, class_index]) - TP
    TN = np.sum(confusion_matrix) - TP - FN - FP

    return TP, TN, FP, FN


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


class ConfusionMatrix:
    def __init__(self, nc, conf=0.75, iou_thres=0.5):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def tp_fp(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class

    def plot(self, normalize=True, save_dir='.', names=()):
        try:
            import seaborn as sn

            array = self.matrix / (
                (self.matrix.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # normalize columns
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.2 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered

                # sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 10}, cmap='Reds', fmt='.2f', square=True,
                #            xticklabels=names + ['background'] if labels else "auto",
                #            yticklabels=names + ['background'] if labels else "auto").set_facecolor((1, 1, 1))

                sn.heatmap(array, cbar=True, annot=True, square=True, fmt=".2f", annot_kws={"size": 10},
                           xticklabels=names + ['background'] if labels else "auto",
                           yticklabels=names + ['background'] if labels else "auto").set_facecolor((1, 1, 1))

            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.jpg', dpi=2000)
            plt.show()
            plt.close()
        except Exception as e:
            print(f'WARNING: ConfusionMatrix plot failure: {e}')

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


# 修改自己的类别
VOC_CLASSES = ('crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches')
# 存放后面生成的 annots.pkl, 主要是标签信息（坐标，类别）
cachedir = '/home/wsjc/Sxkai/new/ANNo/annots.pkl'
# 测试集前缀名文件
imagesetfile = '/home/wsjc/Sxkai/new/EHD-Net/data/NEU-DET/VOC2007/ImageSets/Main/test.txt'
# xml文件夹
annopath = '/home/wsjc/Sxkai/new/EHD-Net/data/NEU-DET/VOC2007/Annotations/{:s}.xml'
# 存放检测结果（坐标，类别，置信度）
predict_txt = '/home/wsjc/Sxkai/new/HUNXIAO/{:s}.txt'

# first load gt 生成标签文件
if not os.path.isdir(cachedir):
    os.mkdir(cachedir)
cachefile = os.path.join(cachedir, "annots.pkl")

# read list of images
with open(imagesetfile, "r") as f:
    lines = f.readlines()
imagenames = [x.strip() for x in lines]

if not os.path.isfile(cachefile):
    # load annots
    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_rec(annopath.format(imagename))
        if i % 100 == 0:
            print("Reading annotation for {:d}/{:d}".format(i + 1, len(imagenames)))
    # save
    print("Saving cached annotations to {:s}".format(cachefile))
    with open(cachefile, "wb") as f:
        pickle.dump(recs, f)
else:
    # load
    with open(cachefile, "rb") as f:
        recs = pickle.load(f)

# 注意修改置信度阈值和IoU阈值
confusion_matrix = ConfusionMatrix(nc=len(VOC_CLASSES), conf=0.2, iou_thres=0.5)
labels = []
for item in recs.values():
    labels.extend([[VOC_CLASSES.index(obj["name"])] + (obj["bbox"]) for obj in item])
labelsn = torch.tensor(labels)

pred = []
for i, cls in enumerate(VOC_CLASSES):  #
    predict_file = predict_txt.format(cls)  # 该文件中所有的框均被预测为该cls类别
    with open(predict_file, "r") as f:
        lines = f.readlines()
    splitlines = [x.strip().split(" ") for x in lines]
    pred += [[float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[1]), i] for line in
             tqdm.tqdm(splitlines)]

predn = torch.tensor(pred)

"""
Arguments:
        predn (Array[N, 6]), x1, y1, x2, y2, conf, class
        labelsn (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        None, updates confusion matrix accordingly
"""
confusion_matrix.process_batch(predn, labelsn)
print(confusion_matrix.matrix)
hunxiao=np.transpose(confusion_matrix.matrix)
class_index = 0
TP, TN, FP, FN = calculate_confusion_matrix_values(hunxiao, class_index)
Rec=TP/(TP+FN)
TNR=TN/(TN+FP)
print(f"Class 0: TP={TP}, TN={TN}, FP={FP}, FN={FN}")
print(f"Class 0: Pre={TP/(TP+FP)}, Rec={TP/(TP+FN)}, TNR={TN/(TN+FP)},Gmean={0.5**(Rec*TNR)}")

# 计算第二类的 TP, TN, FP, FN
class_index = 1
TP, TN, FP, FN = calculate_confusion_matrix_values(hunxiao, class_index)
Rec=TP/(TP+FN)
TNR=TN/(TN+FP)
print(f"Class 1: TP={TP}, TN={TN}, FP={FP}, FN={FN}")
print(f"Class 1: Pre={TP/(TP+FP)}, Rec={TP/(TP+FN)}, TNR={TN/(TN+FP)},Gmean={0.5**(Rec*TNR)}")

confusion_matrix.plot(save_dir="/home/wsjc/Sxkai/new/", names=list(VOC_CLASSES))

