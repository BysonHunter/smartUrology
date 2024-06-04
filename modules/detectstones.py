import argparse
from pathlib import Path
import os
import sys
import PySimpleGUI as sg
import shutil
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from modules import config as cf

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

lang_of_interface, path_to_yolo_weights, default_input_dicom_folder, default_output_folder, current_color_theme, \
    img_count, img_format = cf.read_settings()
tooltips, buttons, list_text, checkbox, main_menu, window_heads, titles = cf.set_language(lang_of_interface)


# print(det)
@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    global c1_array, c5_array, c1, c5, conf_c1, conf_c5, x1_c1, x1_c5, y1_c1, y1_c5
    global c0, c2, c3, c4, conf_c0, conf_c2, conf_c3, conf_c4, x1_c0, x1_c2, x1_c3, x1_c4, y1_c0, y1_c2, y1_c3, y1_c4
    global w1_c0, w1_c2, w1_c3, w1_c4, h1_c0, h1_c2, h1_c3, h1_c4
    global im0
    global line0, xywh_c0, line2, xywh_c2, line3, xywh_c3, line4, xywh_c4, line1, xywh_c1, line5, xywh_c5

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir = Path(project) / name
    (save_dir / 'labels' if save_txt else save_dir)  # .mkdir(parents=True, exist_ok=True)  # make dir
    if not os.path.exists(str(save_dir) + '/labels'):  # mkdir if not exist
        os.makedirs(str(save_dir) + '/labels')

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # create progress bar
    n = len(os.listdir(source))
    current_file = ''
    save_file = ''
    bar_i = 0
    layout_bar = [
        [sg.Text(f'{list_text[15]} {source}')],
        [sg.Text(f'{list_text[16]} {current_file}', key='-FILE-')],
        [sg.Text(f'{list_text[17]} {save_file}', key='-TXTFILE-')],
        [sg.ProgressBar(n, orientation='h', size=(len(list_text[15] + source), 10), border_width=1, key='-PROG-')],
    ]
    # create the progress bar Window
    window_bar = sg.Window('', layout_bar, no_titlebar=False, finalize=True, disable_close=True)

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        predict = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        predict = non_max_suppression(predict, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        im0_000 = ''
        detected_object = False

        # Process predictions
        for i, det in enumerate(predict):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            current_file = p
            # update progress bar
            window_bar['-PROG-'].update(bar_i + 1)
            window_bar['-FILE-'].update(current_file)
            bar_i += 1

            c1 = c5 = c0 = c2 = c3 = c4 = False
            conf_c0 = x1_c0 = y1_c0 = w1_c0 = h1_c0 = 0
            conf_c2 = x1_c2 = y1_c2 = w1_c2 = h1_c2 = 0
            conf_c3 = x1_c3 = y1_c3 = w1_c3 = h1_c3 = 0
            conf_c4 = x1_c4 = y1_c4 = w1_c4 = h1_c4 = 0
            conf_c1 = x1_c1 = y1_c1 = w1_c1 = h1_c1 = 0
            conf_c5 = x1_c5 = y1_c5 = w1_c5 = h1_c5 = 0
            c1_array = []
            c5_array = []

            if len(det):
                # print(f'len det = {len(det)}')
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                '''
                # class names
                names: ['left_kidney' -              0,
                        'stone' -                    1,
                        'right_kidney' -             2,
                        'left_kidney_pieloectasy' -  3,
                        'right_kidney_pieloectasy' - 4,
                        'staghorn_stones' -          5]
                '''

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                    x1, y1, w1, h1, = xywh
                    c = int(cls)
                    if c == 1 and conf >= 0.6:  # stone type stone
                        c1 = True
                        conf_c1 = conf
                        line1 = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        xywh_c1 = xyxy
                        label1 = f'{names[1]} {conf_c1:.2f}'
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line1)).rstrip() % line1 + '\n')
                        annotator.box_label(xywh_c1, label1, color=colors(c, True))
                        # c1_array.append([x1, y1, w1, h1, conf])

                    elif c == 5 and conf >= 0.6:  # stone type staghorn_stones
                        c5 = True
                        conf_c5 = conf
                        line5 = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        xywh_c5 = xyxy
                        label5 = f'{names[5]} {conf_c5:.2f}'
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line5)).rstrip() % line5 + '\n')
                        annotator.box_label(xywh_c5, label5, color=colors(c, True))
                        # c5_array.append([x1, y1, w1, h1, conf])

                    elif c == 0 and \
                            0.55 < x1 < 0.704613 and 0.248879 < y1 < 0.66704 and w1 * h1 > w1_c0 * h1_c0:
                        c0 = True
                        conf_c0 = conf
                        x1_c0 = x1
                        y1_c0 = y1
                        w1_c0 = w1
                        h1_c0 = h1
                        x1y1w1h1 = x1_c0, y1_c0, w1_c0, h1_c0
                        line0 = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        xywh_c0 = xyxy
                    # elif w1 * h1 > w1_c0 * h1_c0:
                    #   conf_c0 = conf
                    #   x1_c0 = x1
                    #   y1_c0 = y1
                    #   w1_c0 = w1
                    #   h1_c0 = h1
                    #   line0 = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                    #   xywh_c0 = xyxy
                    elif c == 3 and \
                            0.55 < x1 < 0.704613 and 0.248879 < y1 < 0.66704 and w1 * h1 > w1_c3 * h1_c3:
                        c3 = True
                        conf_c3 = conf
                        x1_c3 = x1
                        y1_c3 = y1
                        w1_c3 = w1
                        h1_c3 = h1
                        line3 = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        xywh_c3 = xyxy
                    # elif w1 * h1 > w1_c3 * h1_c3:
                    #   conf_c3 = conf
                    #   x1_c3 = x1
                    #   y1_c3 = y1
                    #   w1_c3 = w1
                    #   h1_c3 = h1
                    #   line3 = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                    #   xywh_c3 = xyxy
                    elif c == 2 and \
                            0.2 < x1 < 0.45 and 0.2 < y1 < 0.804613 and w1 * h1 > w1_c2 * h1_c2:
                        c2 = True
                        conf_c2 = conf
                        x1_c2 = x1
                        y1_c2 = y1
                        w1_c2 = w1
                        h1_c2 = h1
                        line2 = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        xywh_c2 = xyxy
                    # elif w1 * h1 > w1_c2 * h1_c2:
                    #   conf_c2 = conf
                    #   x1_c2 = x1
                    #   y1_c2 = y1
                    #   w1_c2 = w1
                    #   h1_c2 = h1
                    #   line2 = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                    #   xywh_c2 = xyxy
                    elif c == 4 and \
                            0.2 < x1 < 0.45 and 0.2 < y1 < 0.804613 and w1 * h1 > w1_c4 * h1_c4:
                        c4 = True
                        conf_c4 = conf
                        x1_c4 = x1
                        y1_c4 = y1
                        w1_c4 = w1
                        h1_c4 = h1
                        line4 = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        xywh_c4 = xyxy
                    # elif w1 * h1 > w1_c4 * h1_c4:
                    #   conf_c4 = conf
                    #   x1_c4 = x1
                    #   y1_c4 = y1
                    #   w1_c4 = w1
                    #   h1_c4 = h1
                    #   line4 = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                    #   xywh_c4 = xyxy

                im0_000 = im0
                # print kidney
                if c0:
                    label0 = f'{names[0]} '  # {conf_c0:.2f}' remove confidence from image
                    with open(f'{txt_path}.txt', 'a') as f:
                        f.write(('%g ' * len(line0)).rstrip() % line0 + '\n')
                    annotator.box_label(xywh_c0, label0, color=colors(c, True))
                    detected_object = True

                if c2:
                    label2 = f'{names[2]}'  # {conf_c2:.2f}' remove confidence from image
                    with open(f'{txt_path}.txt', 'a') as f:
                        f.write(('%g ' * len(line2)).rstrip() % line2 + '\n')
                    annotator.box_label(xywh_c2, label2, color=colors(c, True))
                    detected_object = True

                if c3:
                    label3 = f'{names[3]}'  # {conf_c3:.2f}'remove confidence from image
                    with open(f'{txt_path}.txt', 'a') as f:
                        f.write(('%g ' * len(line3)).rstrip() % line3 + '\n')
                    annotator.box_label(xywh_c3, label3, color=colors(c, True))
                    detected_object = True

                if c4:
                    label4 = f'{names[4]}'  # {conf_c4:.2f}' remove confidence from image
                    with open(f'{txt_path}.txt', 'a') as f:
                        f.write(('%g ' * len(line4)).rstrip() % line4 + '\n')
                    annotator.box_label(xywh_c4, label=label4, color=colors(c, True))
                    detected_object = True

                # print stones!!!!
                '''
                if c1:
                    label = f'{names[1]} {conf_c1:.2f}'
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line1)).rstrip() % line1 + '\n')
                    plot_one_box(xywh_c1, im0_000, label=label, color=colors[1], line_thickness=opt.line_thickness)

                if c5:
                    label = f'{names[5]} {conf_c5:.2f}'
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line5)).rstrip() % line5 + '\n')
                    plot_one_box(xywh_c5, im0_000, label=label, color=colors[5], line_thickness=opt.line_thickness)
                '''

            """
            if len(c1_array) > 0:
                for i in range(len(c1_array)):
                    x1 = c1_array[i][0]
                    y1 = c1_array[i][1]
                    w1 = c1_array[i][2]
                    h1 = c1_array[i][3]

                    if c0:
                        detected_object = True
                        if (x1 > (x1_c0 - w1_c0 / 2)) and (x1 < (x1_c0 + w1_c0 / 2)) and (
                                y1 > (y1_c0 - h1_c0 / 2)) and (
                                y1 < (y1_c0 + h1_c0 / 2)):
                            # print('Камень в левой почке')
                            # with open(txt_path + '.txt', 'a') as f:
                            #    f.write(f'Камень {i+1} в левой почке \n')
                            if opt.save_crop:
                                save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    else:
                        if c3:
                            detected_object = True
                            if (x1 > (x1_c3 - w1_c3 / 2)) and (x1 < (x1_c3 + w1_c3 / 2)) and (
                                    y1 > (y1_c3 - h1_c3 / 2)) and (y1 < (y1_c3 + h1_c3 / 2)):
                                # print('Камень в левой почке')
                                # with open(txt_path + '.txt', 'a') as f:
                                #    f.write(f'Камень {i+1} в левой почке\n')
                                if opt.save_crop:
                                    save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                                                 BGR=True)

                    if c2:
                        detected_object = True
                        if (x1 > (x1_c2 - w1_c2 / 2)) and (x1 < (x1_c2 + w1_c2 / 2)) and (
                                y1 > (y1_c2 - h1_c2 / 2)) and (
                                y1 < (y1_c2 + h1_c2 / 2)):
                            # print('Камень в правой почке')
                            # with open(txt_path + '.txt', 'a') as f:
                            #    f.write(f'Камень {i+1} в правой почке\n')
                            if opt.save_crop:
                                save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    else:
                        if c4:
                            detected_object = True
                            if (x1 > (x1_c4 - w1_c4 / 2)) and (x1 < (x1_c4 + w1_c4 / 2)) and (
                                    y1 > (y1_c4 - h1_c4 / 2)) and (y1 < (y1_c4 + h1_c4 / 2)):
                                #    print('Камень в правой почке')
                                #    with open(txt_path + '.txt', 'a') as f:
                                #        f.write(f'Камень {i+1} в правой почке\n')
                                if opt.save_crop:
                                    save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                                                 BGR=True)

                # print(f'len c5_array = {len(c5_array)}')
            if len(c5_array) > 0:
                for i in range(len(c5_array)):
                    x1 = c5_array[i][0]
                    y1 = c5_array[i][1]
                    w1 = c5_array[i][2]
                    h1 = c5_array[i][3]

                    if c0:
                        detected_object = True
                        if (x1 > (x1_c0 - w1_c0 / 2)) and (x1 < (x1_c0 + w1_c0 / 2)) and (
                                y1 > (y1_c0 - h1_c0 / 2)) and (y1 < (y1_c0 + h1_c0 / 2)):
                            #    print('Камень в левой почке')
                            #    with open(txt_path + '.txt', 'a') as f:
                            #        f.write(f'Камень {i+1} в левой почке\n')
                            if opt.save_crop:
                                save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                                             BGR=True)
                    else:
                        if c3:
                            detected_object = True
                            if (x1 > (x1_c3 - w1_c3 / 2)) and (x1 < (x1_c3 + w1_c3 / 2)) and (
                                    y1 > (y1_c3 - h1_c3 / 2)) and (y1 < (y1_c3 + h1_c3 / 2)):
                                #        print('Камень в левой почке')
                                #        with open(txt_path + '.txt', 'a') as f:
                                #            f.write(f'Камень {i+1} в левой почке\n')
                                if opt.save_crop:
                                    save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                                                 BGR=True)

                    if c2:
                        detected_object = True
                        if (x1 > (x1_c2 - w1_c2 / 2)) and (x1 < (x1_c2 + w1_c2 / 2)) and (
                                y1 > (y1_c2 - h1_c2 / 2)) and (y1 < (y1_c2 + h1_c2 / 2)):
                            #    print('Камень в правой почке')
                            #    with open(txt_path + '.txt', 'a') as f:
                            #        f.write(f'Камень {i+1} в правой почке\n')
                            if opt.save_crop:
                                save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                                             BGR=True)
                    else:
                        if c4:
                            detected_object = True
                            if (x1 > (x1_c4 - w1_c4 / 2)) and (x1 < (x1_c4 + w1_c4 / 2)) and (
                                    y1 > (y1_c4 - h1_c4 / 2)) and (y1 < (y1_c4 + h1_c4 / 2)):
                                #        print('Камень в правой почке')
                                #        with open(txt_path + '.txt', 'a') as f:
                                #            f.write(f'Камень {i+1} в правой почке\n')
                                if opt.save_crop:
                                    save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                                                 BGR=True)
            """

            # Print time (inference + NMS)
            window_bar['-TXTFILE-'].update(f'{s} Done. ({t2 - t1:.3f}s)')
            # print(f'{s} Done. ({t2 - t1:.3f}s)')

            # Stream results
            im0_000 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    if detected_object:
                        cv2.imwrite(save_path, im0_000)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")

    window_bar.close()


def detect_stones(detect_folder, save_conf=True, yolo_weights='weights/kidney_best_191222.pt'):
    # print(yolo_weights)
    if os.path.isdir(detect_folder + '/detect'):
        shutil.rmtree(detect_folder + '/detect')
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=yolo_weights,
                        help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=detect_folder, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[512], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', default=False, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', default=save_conf, action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=detect_folder, help='save results to project/name')
    parser.add_argument('--name', default='detect', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                # kd.detect(opt=opt)
                strip_optimizer(opt.weights)
        else:
            run(**vars(opt))
