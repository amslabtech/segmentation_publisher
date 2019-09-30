# segmentation_publisher

## Requirements
- ROS2 dashing
- Pytorch

## Usage
```
cd YOUR_ROS2_WS
clone this repository
colcon build --symlink-install
source install/local_setup.bash
```

## Node
### segmentation_publisher
- Published Topic
	- /segmented_mage (sensor_msgs/Image)
		- Image labeled in 20 classes
- Subscribed topic
  - /usb_cam/image_raw/compressed (sensor_msgs/CompressedImage)

## Test Video
- test video is available on  [Google Drive](https://drive.google.com/drive/folders/1Tgieyrfuvv3EO0X1CqZwQcaffGdi4bo5?usp=sharing)

##Note
- Please download pretrained model from here.

## Dependencies
- [LEDNet](https://github.com/xiaoyufenfei/LEDNet)


