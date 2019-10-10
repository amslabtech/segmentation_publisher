#!/usr/bin/env python
#! coding utf-8

import rospy
from sensor_msgs.msg import Image as RosImage

import numpy as np
import os
import time
import sys
print(sys.version)
if sys.version_info[0] is not 2:
    sys.path.remove('/opt/ros/' + os.environ['ROS_DISTRO'] + '/lib/python2.7/dist-packages')
sys.path.append(os.path.join(os.path.dirname(__file__), '../LEDNet/train'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../LEDNet/test'))
print(sys.path)

import cv2
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image

import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from lednet import Net
from transform import Relabel, ToLabel, Colorize

def get_cityscapes_color():
    return np.array([
                        [128,  64, 128],
                        [244,  35, 232],
                        [ 70,  70,  70],
                        [102, 102, 156],
                        [190, 153, 153],
                        [153, 153, 153],
                        [250, 170,  30],
                        [220, 220,   0],
                        [107, 142,  35],
                        [152, 251, 152],
                        [ 70, 130, 180],
                        [220,  20,  60],
                        [255,   0,   0],
                        [  0,   0, 142],
                        [  0,   0,  70],
                        [  0,  60, 100],
                        [  0,  80, 100],
                        [  0,   0, 230],
                        [119,  11,  32],
                        [  0,   0,   0]
                    ]).astype("uint8")

class SemanticSegmentation:
    def __init__(self):
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", RosImage, self.image_callback, queue_size=1)
        self.segmented_image_pub = rospy.Publisher("/lednet/segmented_image", RosImage, queue_size=1)
        self.masked_image_pub = rospy.Publisher("/lednet/masked_image", RosImage, queue_size=1)

        self.use_subscribed_images_stamp = True
        if rospy.has_param("USE_SUBSCRIBED_IMAGES_STAMP"):
            self.use_subscribed_images_stamp = rospy.get_param("USE_SUBSCRIBED_IMAGES_STAMP")
        self.WEIGHT_PATH = os.path.join(os.path.dirname(__file__), "model.pth")
        if rospy.has_param("WEIGHT_PATH"):
            self.WEIGHT_PATH = rospy.get_param("WEIGHT_PATH")
        print(self.WEIGHT_PATH)

        self.CLASS_NUM = 20
        self.model = Net(self.CLASS_NUM)
        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()

        model_dict = self.model.state_dict()
        # print(model_dict)
        pretrained_dict = torch.load(self.WEIGHT_PATH)
        # print(pretrained_dict)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print(pretrained_dict)
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        self.model.eval()
        print("=== lednet_ros ===")
        print("waiting for image...")

    def image_callback(self, data):
        try:
            start = time.time()
            print("=== callback ===")
            cv_image = CvBridge().imgmsg_to_cv2(data, "bgr8")
            print(cv_image.shape)
            pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            image = ToTensor()(pil_image)
            image = torch.Tensor(np.array([image.numpy()]))
            input_image = Variable(image)
            input_image = input_image.cuda()
            print('input image shape')
            print(input_image.shape)

            with torch.no_grad():
                output_image = self.model(input_image)
                print('inference time: {0}'.format(time.time() - start) + '[s]')

            # print(output_image.shape)

            label = output_image[0].max(0)[1].byte().cpu().data
            # print(label.shape)
            # print(label)
            # label_color = Colorize()(label.unsqueeze(0))
            label_color = get_cityscapes_color()[label]
            print('colorize time: {0}'.format(time.time() - start) + '[s]')
            # print(label_color.shape)

            label_color = np.asarray(ToPILImage()(label_color))
            print('segmented image shape')
            print(label_color.shape)
            # print(label_color.dtype)

            label_color = cv2.resize(label_color, cv_image.shape[:2][::-1])
            print('output image shape')
            print(label_color.shape)
            pub_seg_image = CvBridge().cv2_to_imgmsg(label_color, "rgb8")
            pub_seg_image.header = data.header
            if not self.use_subscribed_images_stamp:
                    pub_seg_image.header.stamp = rospy.get_rostime()
            self.segmented_image_pub.publish(pub_seg_image)
            masked_image = cv2.addWeighted(cv_image, 0.5, label_color, 0.5, 0)
            pub_masked_image = CvBridge().cv2_to_imgmsg(masked_image, "rgb8")
            pub_masked_image.header = data.header
            if not self.use_subscribed_images_stamp:
                pub_masked_image.header.stamp = rospy.get_rostime()
            self.masked_image_pub.publish(pub_masked_image)

            print('callback time: {0}'.format(time.time() - start) + '[s]')
        except CvBridgeError as e:
            print(e)

    def process(self):
        rospy.init_node('semantic_segmentation', anonymous=True)
        rospy.spin()

if __name__ == '__main__':
    ss = SemanticSegmentation()
    try:
        ss.process()
    except rospy.ROSInterruptException: pass_
