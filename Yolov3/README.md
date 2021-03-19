# Yolov3 on the OpenCV AI Kit

## Introduction

In order to run a YOLOv3 model on the OpenCV AI Kit, the model needs to be converted to a .blob file. This, however, isn't a one-step process. First, the model is converted into a Tensorflow frozen model; then, it's [converted to an OpenVINO Intermediate Representation (IR)](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html). Lastly it's converted to a .blob file using the [online BlobConverter app from Luxonis](http://69.164.214.171:8083/).

## Convert YOLO model to Tensorflow frozen model

You can convert your YOLOv3 model to a Tensorflow frozen model using the [tensorflow-yolo-v3 repository](https://github.com/mystic123/tensorflow-yolo-v3).

1. Clone the repository:
    ```bash
    git clone https://github.com/mystic123/tensorflow-yolo-v3.git
    cd tensorflow-yolo-v3
    ```
2. (Optional) Checkout to the commit that the conversion was tested on:
    ```bash
    git checkout ed60b90
    ```
3. Run the converter
    ```bash
    python3 convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3-tiny.weights --tiny
    ```

## Convert Tensorflow model to OpenVINO IR

After you have an exported model (.pb file), you can convert it into an OpenVINO IR model using the [OpenVINO model optimizer](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html):

1. Installation
    ```python
    %%time
    %%capture
    ## install tools. Open Vino takes some time to download: 10-15 min sometimes.
    !sudo apt-get install -y pciutils cpio
    !sudo apt autoremove
    ## downnload installation files
    !wget https://registrationcenter-download.intel.com/akdlm/irc_nas/17504/l_openvino_toolkit_p_2021.2.185.tgz
    path = "l_openvino_toolkit_p_2021.2.185.tgz"
    ## install openvino
    !tar xf "{path}"
    %cd l_openvino_toolkit_p_2021.2.185/
    !./install_openvino_dependencies.sh && \
        sed -i 's/decline/accept/g' silent.cfg && \
        ./install.sh --silent silent.cfg
    %cd /opt/intel/openvino_2021.2.185/deployment_tools/model_optimizer/install_prerequisites/
    !./install_prerequisites_tf.sh
    ```
2. Create config file
    yolo_v3_tiny.json:
    ```json
    [
        {
            "id": "TFYOLOV3",
            "match_kind": "general",
            "custom_attributes": {
            "classes": 80,
            "anchors": [10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319],
            "coords": 4,
            "num": 6,
            "masks": [[3, 4, 5], [0, 1, 2]],
            "entry_points": ["detector/yolo-v3-tiny/Reshape", "detector/yolo-v3-tiny/Reshape_4"]
            }
        }
    ]
    ```
3. Convert model
    ```
    output_dir = '/content/yolov3_tiny'

    !source /opt/intel/openvino_2021.2.185/bin/setupvars.sh && \
        python /opt/intel/openvino_2021.2.185/deployment_tools/model_optimizer/mo.py \
        --input_model /content/darknet/tensorflow-yolo-v3/frozen_darknet_yolov3_model.pb \
        --tensorflow_use_custom_operations_config /content/darknet/yolo_v3_tiny.json \
        --batch 1 \
        --data_type FP16 \
        --reverse_input_channel \
        --output_dir {output_dir}
    ```

## Compile the IR model to a .blob for use on DepthAI modules/platform

After converting your model into OpenVINO IR format, you can compile it to a .blob file so it can be used on the OpenCV AI Kit using the [online BlobConverter app from Luxonis](http://69.164.214.171:8083/), which can be used with the following code.

```python
import requests

url = "http://69.164.214.171:8083/compile"  # change if running against other URL

payload = {
    'compiler_params': '-ip U8 -VPU_NUMBER_OF_SHAVES 8 -VPU_NUMBER_OF_CMX_SLICES 8',
    'compile_type': 'myriad'
}
files = [
    ('definition', open(f'{output_dir}/frozen_darknet_yolov3_model.xml', 'rb')),
    ('weights', open(f'{output_dir}/frozen_darknet_yolov3_model.bin', 'rb'))
]
params = {
    'version': '2021.1',  # OpenVINO version, can be "2021.1", "2020.4", "2020.3", "2020.2", "2020.1", "2019.R3"
}

response = requests.post(url, data=payload, files=files, params=params)

with open(f'{output_dir}/model.blob', 'wb') as f:
  f.write(response.content)
```

## Use new model on the OAK Device

To use the model, create a directory called models and then place the model.blob file inside it. Then go inside the [run_yolo_model.py](run_yolo_model.py) file and replace the label_map array with your custom labels.

After that, you can run the [run_yolo_model.py script](run_yolo_model.py).

```bash
python run_yolo_model.py
```