# based on https://github.com/luxonis/depthai-tutorials/tree/master/1-hello-world
from pathlib import Path
import json

import cv2  # opencv - display the video stream
import depthai  # access the camera and its data packets

device = depthai.Device('', False)

config = {
    'streams': ['previewout', 'metaout'],
    'ai': {
        "blob_file": str(Path('./model/model.blob').resolve().absolute()),
        "blob_file_config": str(Path('./model/model.json').resolve().absolute()),
    }
}

# Create the pipeline using the 'previewout' stream, establishing the first connection to the device.
pipeline = device.create_pipeline(config=config)

# Retrieve model class labels from model config file.
model_config_file = config['ai']['blob_file_config']
mcf = open(model_config_file)
model_config_dict = json.load(mcf)
labels = model_config_dict['mappings']['labels'] if 'mappings' in model_config_dict else None

if pipeline is None:
    raise RuntimeError('Pipeline creation failed!')

detections = []

while True:
    # Retrieve data packets from the device.
    # A data packet contains the video frame data.
    nnet_packets, data_packets = pipeline.get_available_nnet_and_data_packets()

    for nnet_packet in nnet_packets:
        detections = list(nnet_packet.getDetectedObjects())

    for packet in data_packets:
        # By default, DepthAI adds other streams (notably 'meta_2dh'). Only process `previewout`.
        if packet.stream_name == 'previewout':
            data = packet.getData()
            # change shape (3, 300, 300) -> (300, 300, 3)
            data0 = data[0, :, :]
            data1 = data[1, :, :]
            data2 = data[2, :, :]
            frame = cv2.merge([data0, data1, data2])

            img_h = frame.shape[0]
            img_w = frame.shape[1]

            for detection in detections:
                pt1 = int(detection.x_min * img_w), int(detection.y_min * img_h)
                pt2 = int(detection.x_max * img_w), int(detection.y_max * img_h)
                print(int(detection.label))
                label = labels[int(detection.label)] if labels != None else int(detection.label)
                score = int(detection.confidence * 100)
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
                cv2.putText(frame, str(score) + '% ' + str(label), (pt1[0] + 2, pt1[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('previewout', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# The pipeline object should be deleted after exiting the loop. Otherwise device will continue working.
# This is required if you are going to add code after exiting the loop.
del device
