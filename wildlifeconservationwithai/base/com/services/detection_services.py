import os
import platform
import sys
import cv2
import time
import torch
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
import torch.backends.cudnn as cudnn
import winsound
from datetime import datetime
from conda_build import windows

from base.com.ai_module.trackbleobject import TrackableObject
from base.com.ai_module.utils.general import set_logging
from base.com.ai_module.models.common import DetectMultiBackend
from base.com.ai_module.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from base.com.ai_module.utils.general import (LOGGER, Profile, check_file, check_img_size,
                           check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args,
                           scale_coords, strip_optimizer, xyxy2xywh)
from base.com.ai_module.utils.plots import Annotator, colors, save_one_box
from base.com.ai_module.utils.torch_utils import select_device, time_sync
from base.com.vo.detection_images_vo import DetectionImagesVO
from base.com.dao.detection_dao import DetectionDAO



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# ---------------Object Tracking---------------
import skimage
from base.com.ai_module.deepsort.sort import *

# -----------Object Blurring-------------------
blurratio = 40

# .................. Tracker Functions .................
'''Computer Color for every box and track'''
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

# -----------------------
trackers = []
trackableObjects = {}

totalDown = 0
totalUp = 0
#------------------------------------


# -----------------------
def Alert(im0):
    output_image_path=r'base\static\adminResources\output_image'
    winsound.Beep(1000, 400)
    # Save the photo and timestamp in the database
    timestamp = datetime.now()
    detection_time = timestamp.strftime('%H:%M:%S')
    detection_date = timestamp.strftime('%D')
    image_name = f"\lion_detected-{timestamp.strftime('%Y%m%d-%H%M%S')}.jpg"
    image_path = output_image_path + image_name
    print(output_image_path)
    print(image_path)
    cv2.imwrite(image_path, im0)

    # code to save event in database

    if os.path.exists(image_path):
        print('Image Saved')
        detection_image_vo = DetectionImagesVO()
        detection_dao = DetectionDAO()

        image_path = image_path.replace('base\static', '\static')

        detection_image_vo.image_name = image_name
        detection_image_vo.image_file_path = image_path
        detection_image_vo.detection_time = detection_time
        detection_image_vo.detection_date = detection_date

        detection_dao.add_detection_images(detection_image_vo)
    else:
        print('Image Not Exists')



def compute_color_for_labels(label):
    color = [int(int(p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


"""" Calculates the relative bounding box from absolute pixel values. """


def bbox_rel(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


"""Function to Draw Bounding boxes"""


def draw_boxes(img, bbox, identities=None, categories=None,
               names=None, color_box=None, offset=(0, 0)):
    print(bbox)
    for i, box in enumerate(bbox):
        print(">>>>>>>>>>>>>>>>>>>>>>", i, box)
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        data = (int((box[0] + box[2]) / 2), (int((box[1] + box[3]) / 2)))
        label = str(id)

        if color_box:
            color = compute_color_for_labels(id)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        1)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 191, 0), -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        [255, 255, 255], 1)
            cv2.circle(img, data, 3, color, -1)
        else:
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 191, 0), 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 191, 0), -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        [255, 255, 255], 1)
            cv2.circle(img, data, 3, (255, 191, 0), -1)
    return img


# ..............................................................................


@torch.no_grad()
def run(
        weights=r'D:\PROJECT\projectworkspace\wildlifeconservationwithai\base\static\weights\best.pt',
        source=r'D:\PROJECT\projectworkspace\wildlifeconservationwithai\base\static\adminResources\input_videos\video-1.mp4',
        data=r'D:\PROJECT\projectworkspace\wildlifeconservationwithai\base\com\ai_module\data.yaml',
        imgsz=(640, 640), conf_thres=0.60, iou_thres=0.45,
        max_det=1000, device='cpu', view_img=True,
        save_txt=False, save_conf=False, save_crop=False,
        nosave=False, classes=None, agnostic_nms=False,
        augment=False, visualize=False, update=False,
        project=r'D:\PROJECT\projectworkspace\wildlifeconservationwithai\base\static\adminResources\output_video',
        name='',
        exist_ok=True, line_thickness=2, hide_labels=False,
        hide_conf=False, half=False, dnn=False, display_labels=False,
        blur_obj=False, color_box=False, ):
    save_img = not nosave and not source.endswith('.txt')

    # .... Initialize SORT ....
    sort_max_age = 5
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)
    track_color_id = 0
    # .........................

    webcam = source.isnumeric() or source.endswith(
        '.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                          exist_ok=True)

    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)

    half &= (pt or jit or onnx or engine) and device.type != 'cpu'
    if pt or jit:
        model.model.half() if half else model.model.float()

    if webcam:
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1
    vid_path, vid_writer = [None] * bs, [None] * bs

    t0 = time.time()

    dt, seen = [0.0, 0.0, 0.0], 0
    # ----------------------------------
    # multi onject up down variables
    listDet = ['person', 'bicycle', 'car', 'motorcycle']
    rects = []
    labelObj = []

    totalDownPerson = 0
    totalDownBicycle = 0
    totalDownCar = 0
    totalDownMotor = 0
    totalDownBus = 0
    totalDownTruck = 0
    totalRightLion = 0

    totalLeftLion = 0
    totalUpPerson = 0
    totalUpBicycle = 0
    totalUpCar = 0
    totalUpMotor = 0
    totalUpBus = 0
    totalUpTruck = 0
    # ----------------------------------
    for path, im, im0s, vid_cap, s in dataset:

        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem,
                                   mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes,
                                   agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + (
                '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness,
                                  example=str(names))


            middle = im0.shape[0] // 2
            height = im0.shape[1]
            width = im0.shape[1]
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4],
                                          im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if blur_obj:
                        crop_obj = im0[int(xyxy[1]):int(xyxy[3]),
                                   int(xyxy[0]):int(xyxy[2])]
                        blur = cv2.blur(crop_obj, (blurratio, blurratio))
                        im0[int(xyxy[1]):int(xyxy[3]),
                        int(xyxy[0]):int(xyxy[2])] = blur
                    else:
                        continue
                # ..................USE TRACK FUNCTION....................
                # pass an empty array to sort
                dets_to_sort = np.empty((0, 6))

                # NOTE: We send in detected object class too
                for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort,
                                              np.array([x1, y1, x2, y2,
                                                        conf, detclass])))

                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks = sort_tracker.getTrackers()

                # draw boxes for visualizationq
                if len(tracked_dets) > 0:
                    bbox_xyxy = tracked_dets[:, :4]
                    print(bbox_xyxy)
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    for cord in tracked_dets:
                        # print("cord ->", cord)
                        x_c = (cord[0] + [2]) / 2
                        print(x_c)
                        y_c = (cord[1] + cord[3]) / 2
                        print(y_c)
                        to = trackableObjects.get(cord[8], None)
                        if to is None:
                            to = TrackableObject(cord[8], (x_c, y_c))
                            print("TOOOO", to)
                        else:
                            y = [c[1] for c in to.centroids]
                            direction = y_c - np.mean(y)
                            to.centroids.append((x_c, y_c))
                            if not to.counted:
                                idx = int(cord[4])
                                print("idx", type(idx))
                                if direction < 0 and middle > y_c > height // 2.3:
                                    # if direction < 0 and (height // 1.7 and width // 2) < y_c > (height // 2 and width // 2.3):

                                    # print("labelObj",labelObj)
                                    if (idx == 0):
                                        totalLeftLion += 1
                                        to.counted = True
                                        Alert(im0)

                                elif direction > 0 and middle < y_c < height // 1.8:
                                    # elif direction > 0 and y_c < (height //1.7 and width //2.7):

                                    if (idx == 0):
                                        totalRightLion += 1
                                        to.counted = True
                                        Alert(im0)

                                if middle >= (width/2)-5 and middle >= (width/2)+5:
                                    print("lion Crosswd Middle line")


                        trackableObjects[cord[8]] = to

                    draw_boxes(im0, bbox_xyxy, identities, categories, names,
                               color_box)

            # start_point = (0, int(height // 2.3))
            # start_point_middle = (0, int(middle))
            # start_point_below = (0, int(height // 1.8))


            start_point = (int(height // 2.3),0)
            start_point_middle = ( int(middle),0)
            start_point_below = (int(height // 1.8),0)

            end_point = (int(width // 2.3),int(im0.shape[0]) )
            end_point_middle = ( int(middle),int(im0.shape[0]))
            end_point_below = (int(width // 1.8),int(im0.shape[0]) )


            # end_point = (int(im0.shape[1]), int(height // 2.3))
            # end_point_middle = (int(im0.shape[1]), int(middle))
            # end_point_below = (int(im0.shape[1]), int(height // 1.8))

            # color = (255,0,0)
            thickness = 15
            # cv2.line(im0,(0,int(height//2)),(int(width//2.4),0),thickness)
            cv2.line(im0, start_point, end_point, thickness)
            cv2.line(im0, start_point_middle, end_point_middle, thickness)
            cv2.line(im0, start_point_below, end_point_below, thickness)

            color = (255, 0, 0)

            cv2.putText(im0, 'Right :' + str(totalRightLion),
                        (int(width * 0.6), int(height * 0.05)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(im0, 'Left : ' + str(totalLeftLion),
                        (int(width * 0.02), int(height * 0.05)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path,
                                                     cv2.VideoWriter_fourcc(
                                                         *'h264'), fps, (w, h))
                    vid_writer.write(im0)
        print("Frame Processing!")
    print("Video Exported Success")

    if update:
        strip_optimizer(weights)

    if vid_cap:
        vid_cap.release()


def main():
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars())


if __name__ == "__main__":
    main()
