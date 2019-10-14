"""
This code is used to batch detect images in a folder.
"""
import argparse
import os
import sys

import cv2

from vision.ssd.config.fd_config import define_img_size
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor


parser = argparse.ArgumentParser(description='detect_imgs')

parser.add_argument('--net_type', default="mb_tiny_RFB_fd", type=str,
                    help='The network architecture ,optional:1. mb_tiny_RFB_fd (higher precision) or 2.mb_tiny_fd (faster)')
parser.add_argument('--input_size', default=640, type=int,help='optional value 128/160/320/480/640/1280')
parser.add_argument('--threshold', default=0.9, type=float )
parser.add_argument('--candidate_size', default=1500, type=int,  help='nms candidate size')
parser.add_argument('--path', default="imgs", type=str, help='imgs dir')
parser.add_argument('--test_device', default="cuda:0", type=str,help='cuda:0 or cpu')
args = parser.parse_args()
define_img_size(args.input_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'


label_path = "./models/voc-model-labels.txt"
test_device = args.test_device

class_names = [name.strip() for name in open(label_path).readlines()]
if args.net_type == 'mb_tiny_fd':
    model_path = "models/pretrained/Mb_Tiny_FD_train_input_320.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=args.candidate_size, device=test_device)
elif args.net_type == 'mb_tiny_RFB_fd':
    model_path = "models/pretrained/Mb_Tiny_RFB_FD_train_input_320.pth"
    # model_path = "models/pretrained/Mb_Tiny_RFB_FD_train_input_640.pth"
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=args.candidate_size, device=test_device)
else:
    print("The net type is wrong!")
    sys.exit(1)
net.load(model_path)

vc = cv2.VideoCapture(0)  # 读入视频文件
while True:  # 循环读取视频帧
    rval, orig_image = vc.read()

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, args.candidate_size / 2, args.threshold)
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        # label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
        label = f"{probs[i]:.2f}"
        cv2.putText(orig_image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(orig_image, str(boxes.size(0)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("sdf",orig_image)
    cv2.waitKey(1)

    # print(f"Found {len(probs)} faces. The output image is {result_path}")

