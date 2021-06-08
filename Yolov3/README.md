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
    import os
    from urllib.parse import urlparse

    ## install tools. Open Vino takes some time to download - it's ~400MB
    !sudo apt-get install -y pciutils cpio
    !sudo apt autoremove

    ## downnload installation files
    url = "https://registrationcenter-download.intel.com/akdlm/irc_nas/17662/l_openvino_toolkit_p_2021.3.394.tgz"
    !wget {url}

    ## Get the name of the tgz
    parsed = urlparse(url)
    openvino_tgz = os.path.basename(parsed.path)
    openvino_folder = os.path.splitext(openvino_tgz)[0]

    ## Extract & install openvino
    !tar xf {openvino_tgz}
    %cd {openvino_folder}
    !./install_openvino_dependencies.sh && \
        sed -i 's/decline/accept/g' silent.cfg && \
        ./install.sh --silent silent.cfg
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

    # Get openvino installation path
    openvino = !find /opt/intel -type d -name openvino*

    !python -mpip install -r {openvino[0]}/deployment_tools/model_optimizer/requirements.txt

    !source {openvino[0]}/bin/setupvars.sh && \
        python {openvino[0]}/deployment_tools/model_optimizer/mo.py \
        --input_model /content/darknet/tensorflow-yolo-v3/frozen_darknet_yolov3_model.pb \
        --tensorflow_use_custom_operations_config /content/darknet/yolo_v3_tiny.json \
        --batch 1 \
        --data_type FP16 \
        --reverse_input_channel \
        --output_dir {output_dir}
    ```

## Compile the IR model to a .blob for use on DepthAI modules/platform

After converting your model into OpenVINO IR format, you can compile it to a .blob file so it can be used on the OpenCV AI Kit using the [online BlobConverter app from Luxonis](https://github.com/luxonis/blobconverter), which can be used with the following code.

```python
binfile = f'{output_dir}/frozen_darknet_yolov3_model.bin'
xmlfile = f'{output_dir}/frozen_darknet_yolov3_model.xml'

!python -m pip install blobconverter

import blobconverter
blob_path = blobconverter.from_openvino(
    xml=xmlfile,
    bin=binfile,
    data_type="FP16",
    shaves=5,
)
from google.colab import files
files.download(blob_path) 
```

## Use new model on the OAK Device

To use the model, create a directory called models and then place the model.blob file inside it. Then go inside the [run_yolo_model.py](run_yolo_model.py) file and replace the label_map array with your custom labels.

After that, you can run the [run_yolo_model.py script](run_yolo_model.py).

```bash
python run_yolo_model.py
```