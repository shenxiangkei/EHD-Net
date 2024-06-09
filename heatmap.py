import argparse
import cv2
import numpy as np
import os
import torch
import tqdm
from detectron2.data.detection_utils import read_image
import time
from detectron2.utils.logger import setup_logger

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    # def run_on_image(self, image):
    #     """
    #     Args:
    #         image (np.ndarray): an image of shape (H, W, C) (in BGR order).
    #             This is the format used by OpenCV.
    #
    #     Returns:
    #         predictions (dict): the output of the model.
    #         vis_output (VisImage): the visualized image output.
    #     """
    #     vis_output = None
    #     predictions = self.predictor(image)
    #     # Convert image from OpenCV BGR format to Matplotlib RGB format.
    #     image = image[:, :, ::-1]
    #     visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
    #     if "panoptic_seg" in predictions:
    #         panoptic_seg, segments_info = predictions["panoptic_seg"]
    #         vis_output = visualizer.draw_panoptic_seg_predictions(
    #             panoptic_seg.to(self.cpu_device), segments_info
    #         )
    #     else:
    #         if "sem_seg" in predictions:
    #             vis_output = visualizer.draw_sem_seg(
    #                 predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
    #             )
    #         if "instances" in predictions:
    #             instances = predictions["instances"].to(self.cpu_device)
    #             vis_output = visualizer.draw_instance_predictions(predictions=instances)
    #
    #     return predictions, vis_output

    # 修改可视化需要的文件predict.py中的run_on_image,对相应代码进行注释
#####XIUGAI!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def run_on_image(self, image):

        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        # image = image[:, :, ::-1]
        # visualizer = Visualizer(image, self.metadata,
        # instance_mode=self.instance_mode)
        # if "panoptic_seg" in predictions:
        #     panoptic_seg, segments_info = predictions["panoptic_seg"]
        #     vis_output = visualizer.draw_panoptic_seg_predictions(
        #         panoptic_seg.to(self.cpu_device), segments_info
        #     )
        # else:
        #     if "sem_seg" in predictions:
        #         vis_output = visualizer.draw_sem_seg(
        #             predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
        #         )
        #     if "instances" in predictions:
        #         instances = predictions["instances"].to(self.cpu_device)
        #         vis_output =
        # visualizer.draw_instance_predictions(predictions=instances)

        return predictions
        # return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5


def setup_cfg(args):  # 获取cfg，并合并，不用细看，和demo.py中的一样
    # load config from file and command-line arguments
    from detectron2.config import get_cfg
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="",  # 此处是配置文件，在config下选择你的yaml文件
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", default='/home/wsjc/Sxkai/new/EHD-Net/data/NEU-MFL/VOC2007/test/', nargs="+",
                        help="A list of space separated input images")  # 图片文件夹路径，目前只支持图片输入，
    # 要输入视频或者调用摄像头，可以自行修改代码
    parser.add_argument(
        "--output",
        default='',  # 输出文件夹路径
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,  # 置信度阈值
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)

    # 1*256*200*256 # feat的维度要求，四维
    feature_map = feature_map.detach()

    # 1*256*200*256->1*200*256
    heatmap = feature_map[:, 0, :, :] * 0
    for c in range(feature_map.shape[1]):
        heatmap += feature_map[:, c, :, :]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap


def draw_feature_map(img_path, save_dir):
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    logger = setup_logger()
    logger.info("Arguments: " + str(args))


    demo = VisualizationDemo(cfg)
    for imgs in tqdm.tqdm(os.listdir(img_path)):
        img = read_image(os.path.join(img_path, imgs), format="BGR")
        start_time = time.time()
        predictions = demo.run_on_image(img)  # 后面需对网络输出做一定修改，
        # 会得到一个字典P3-P7的输出
        logger.info(
            "{}: detected in {:.2f}s".format(
                imgs, time.time() - start_time))
        i = 0
        for featuremap in list(predictions.values()):
            heatmap = featuremap_2_heatmap(featuremap)
            # 200*256->512*640
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的
            # 大小调整为与原始图像相同
            heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
            # 512*640*3
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原
            # 始图像
            superimposed_img = heatmap * 0.7 + 0.3 * img  # 热力图强度因子，修改参数，得到合适的热力图

            cv2.imwrite(os.path.join(save_dir, imgs + str(i) + '.jpg'),
                    superimposed_img)  # 将图像保存
            i = i + 1



from argparse import ArgumentParser


def main():
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    draw_feature_map(args.input, args.output)


if __name__ == '__main__':
    main()