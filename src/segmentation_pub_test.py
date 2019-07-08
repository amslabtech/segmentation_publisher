import numpy as np
import torch
import os
import importlib
import sys
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt

from PIL import Image as PIL_Image
from argparse import ArgumentParser
import cv2

from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from lednet import Net

from transform import Relabel, ToLabel, Colorize

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage, CameraInfo

NUM_CLASSES = 20

def load_state(model, state_dict):

    my_state = model.state_dict()

    for name, param in state_dict.items():
        if name not in my_state:
            continue
        my_state[name].copy_(param)

    return model

class Segmentation(Node):
    def __init__(self):
        super().__init__('segmentation_publisher')
        self.bridge = CvBridge()
        self.pub_seg = self.create_publisher(Image, '/recognition/segmentation')
        #self.pub_seg = self.create_publisher(Image, '/amsl/demo/segmentation')
        self.sub = self.create_subscription(CompressedImage, '/usb_cam/image_raw/compressed', self.callback)
        #self.sub = self.create_subscription(Image, '/cam/custom_camera/image_raw', self.callback)

        self.weightspath = "/model_001/model_best.pth"
        self.model = Net(NUM_CLASSES)
        self.model = torch.nn.DataParallel(self.model)

        self.model = self.model.cuda()

        self.model = load_state(self.model, torch.load(self.weightspath))

        self.model.eval()
        print("Ready")

    def callback(self, oimg):

        try:
            oimg_b = bytes(oimg.data)
            np_arr = np.fromstring(oimg_b, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            #img = self.bridge.imgmsg_to_cv2(oimg, "bgr8")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_size = img.shape
            image = PIL_Image.fromarray(img)
            image = image.resize((1024,512),PIL_Image.NEAREST)

            image = ToTensor()(image)
            image = torch.Tensor(np.array([image.numpy()]))

            image = image.cuda()
            
            input_image = Variable(image)
            
            with torch.no_grad():
                output_image = self.model(input_image)
            
            label = output_image[0].max(0)[1].byte().cpu().data
            label_color = Colorize()(label.unsqueeze(0))
            label_pub = ToPILImage()(label_color)
            label_pub = label_pub.resize((img_size[1],img_size[0]),PIL_Image.NEAREST)
            label_pub = np.asarray(label_pub)
            plt.imshow(label_pub)
            plt.pause(0.001)
            self.pub_seg.publish(self.bridge.cv2_to_imgmsg(label_pub, "bgr8"))
     
        except CvBridgeError as e:
            print(e)

def main(args=None):
    
    rclpy.init()
    segmentation = Segmentation()

    try:
        rclpy.spin(segmentation)

    finally:
        if segmentation not in locals():
            segmentation.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
