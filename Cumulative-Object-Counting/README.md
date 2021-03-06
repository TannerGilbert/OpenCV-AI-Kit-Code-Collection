# Cumulative Object Counting

![cumulative object counting](https://raw.githubusercontent.com/TannerGilbert/Tensorflow-2-Object-Counting/master/doc/cumulative_object_counting.PNG)

## Usage

```
usage: cumulative_object_counting.py [-h] -m MODEL [-v VIDEO_PATH] [-roi ROI_POSITION] [-a] [-sh] [-sp SAVE_PATH] [-s]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        File path of .blob file. (default: None)
  -v VIDEO_PATH, --video_path VIDEO_PATH
                        Path to video. If empty OAK-RGB camera is used. (default='') (default: )
  -roi ROI_POSITION, --roi_position ROI_POSITION
                        ROI Position (0-1) (default: 0.5)
  -a, --axis            Axis for cumulative counting (default=x axis) (default: True)
  -sh, --show           Show output (default: True)
  -sp SAVE_PATH, --save_path SAVE_PATH
                        Path to save the output. If None output won't be saved (default: )
  -s, --sync            Sync RGB output with NN output (default: False)
```

Camera example:
```
python cumulative_object_counting.py -m mobilenet-ssd/mobilenet-ssd.blob
```

Video example:
```
python cumulative_object_counting.py -m mobilenet-ssd/mobilenet-ssd.blob -v <path to video>
```

## Inspired by / Based on

This project is based on my [Tensorflow 2 Object Counting repository](https://github.com/TannerGilbert/Tensorflow-2-Object-Counting), which in turn is inspired by [OpenCV People Counter](https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/) and the [tensorflow_object_counting_api](https://github.com/ahmetozlu/tensorflow_object_counting_api).