# segmentation_publisher

## Requirements
- ROS2 Crystal
- Pytorch

## Attention
- "camera_info" is Unimplemented

## Node
### segmentation_publisher
- Published Topic
	- /recognition/segmentation (sensor_msgs/Image)
		- Image labeled in 20 classes
- Subscribed topic
  - /usb_cam/image_raw/compressed (sensor_msgs/CompressedImage)

##Test Video
- "test video was uploaded in [Google Drive](https://drive.google.com/drive/folders/1Tgieyrfuvv3EO0X1CqZwQcaffGdi4bo5?usp=sharing)"

##Dependencies
- [LEDNet](https://github.com/xiaoyufenfei/LEDNet)

