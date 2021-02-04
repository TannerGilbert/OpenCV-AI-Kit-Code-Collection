# Tensorflow OD API models on the OpenCV AI Kit

[![TensorFlow 1.15](https://img.shields.io/badge/TensorFlow-1.15-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v1.15.0)

![object_detection_result](doc/object_detection_result.PNG)

## Introduction

In order to run a Tensorflow Object Detection API model on the OpenCV AI Kit, the Tensorflow model needs to be converted to a .blob file. For this, we'll first convert the Tensorflow model to an OpenVINO IR model, and then we'll convert the IR model to a .blob file.

## Convert Tensorflow model to OpenVINO IR

To get started, you'll first need a model. For this guide, you can either use a pre-trained model from the [Tensorflow Model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md), or train your own custom model as described in [one of my other Github repositories](https://github.com/TannerGilbert/Tensorflow-Lite-Object-Detection-with-the-Tensorflow-Object-Detection-API/tree/v1). 

> *Note:* Make sure to use Tensorflow 1 as Tensorflow 2 doesn't seem to work yet.

After you have an exported model (.pb file), you can convert it into an OpenVINO IR model using the  [OpenVINO model optimizer](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html):

1. Installation
    ```python
    %%time
    %%capture
    ## install tools. Open Vino takes some time to download: 10-15 min sometimes.
    !sudo apt-get install -y pciutils cpio
    !sudo apt autoremove
    ## download installation files
    !wget http://registrationcenter-download.intel.com/akdlm/irc_nas/16345/l_openvino_toolkit_p_2020.1.023.tgz
    path = 'l_openvino_toolkit_p_2020.1.023.tgz'
    # path = "/content/software/Intel OpenVINO 2019 R3.1/l_openvino_toolkit_p_2019.3.376.tgz"
    ## install openvino
    !tar xf '{path}'
    %cd l_openvino_toolkit_p_2020.1.023/
    !./install_openvino_dependencies.sh && \
        sed -i 's/decline/accept/g' silent.cfg && \
        ./install.sh --silent silent.cfg
    ```
2. Convert model
    ```
    output_dir = '/content/output'

    %cd '/content/ssd_mobilenet_v2_coco_2018_03_29/'
    !source /opt/intel/openvino/bin/setupvars.sh && \
        python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
        --input_model frozen_inference_graph.pb \
        --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json \
        --tensorflow_object_detection_api_pipeline_config pipeline.config \
        --reverse_input_channels \
        --output_dir {output_dir} \
        --data_type FP16  
    ```

## Compile the IR model to a .blob for use on DepthAI modules/platform

After converting your model into OpenVINO IR format, you can compile it to a .blob file so it can be used on the OpenCV AI Kit using the [online BlobConverter app from Luxonis](http://69.164.214.171:8083/), which can be used with the following code.

```python
import requests

url = 'http://69.164.214.171:8083/compile'  # change if running against other URL

payload = {
    'compiler_params': '-ip U8 -VPU_MYRIAD_PLATFORM VPU_MYRIAD_2480 -VPU_NUMBER_OF_SHAVES 4 -VPU_NUMBER_OF_CMX_SLICES 4',
    'compile_type': 'myriad'
}
files = {
    'definition': open(f'{output_dir}/frozen_inference_graph.xml', 'rb'),
    'weights': open(f'{output_dir}/frozen_inference_graph.bin', 'rb')
}
params = {
    'version': '2020.1',  # OpenVINO version, can be "2021.1", "2020.4", "2020.3", "2020.2", "2020.1", "2019.R3"
}

response = requests.post(url, data=payload, files=files, params=params)

with open(f'{output_dir}/model.blob', 'wb') as f:
  f.write(response.content)
```

## Use new model on the OAK Device

To use the model, just replace the model.blob file inside the model directory with your custom model and then change the labels inside the model.json file to fit your model.

> Note: For whatever reason, I needed to at a null label as the first element

Example:
```json
{
    "NN_config": {
        "output_format" : "detection",
        "NN_family" : "mobilenet",
        "confidence_threshold" : 0.5
    },
    "mappings": {
        "labels": [
            "null",
            "Raspberry_Pi_3",
            "Arduino_Nano",
            "ESP8266",
            "Heltec_ESP32_Lora"
        ]
    }
}
```

After following the above steps, you can run the model by running the `run_object_detection.py` script.

```
python run_object_detection.py
```

![object_detection_result](doc/object_detection_result.PNG)