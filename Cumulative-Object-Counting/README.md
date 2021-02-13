# Cumulative Object Counting

![cumulative object counting](https://raw.githubusercontent.com/TannerGilbert/Tensorflow-2-Object-Counting/master/doc/cumulative_object_counting.PNG)

## Usage

```
usage: cumulative_object_counting.py [-h] -m MODEL -c CONFIG [-roi ROI_POSITION] [-a] [-sh]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        File path of .blob file. (default: None)
  -c CONFIG, --config CONFIG
                        File path of config file. (default: None)
  -roi ROI_POSITION, --roi_position ROI_POSITION
                        ROI Position (0-1) (default: 0.6)
  -a, --axis            Axis for cumulative counting (default=x axis) (default: True)
  -sh, --show           Show output (default: True)
```

Example:
```
python cumulative_object_counting.py -m mobilenet-ssd/mobilenet-ssd.blob -c mobilenet-ssd/mobilenet-ssd.json
```

## Inspired by / Based on

This project is based on my [Tensorflow 2 Object Counting repository](https://github.com/TannerGilbert/Tensorflow-2-Object-Counting), which in turn is inspired by [OpenCV People Counter](https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/) and the [tensorflow_object_counting_api](https://github.com/ahmetozlu/tensorflow_object_counting_api).