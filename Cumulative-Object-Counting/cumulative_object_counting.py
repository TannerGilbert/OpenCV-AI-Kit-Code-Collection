from pathlib import Path
import json
import cv2
import argparse
import numpy as np
import depthai

from trackable_object import TrackableObject
from centroid_tracker import CentroidTracker


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', type=str, required=True, help='File path of .blob file.')
    parser.add_argument('-c', '--config', type=str, required=True, help='File path of config file.')
    parser.add_argument('-roi', '--roi_position', type=float, default=0.6, help='ROI Position (0-1)')
    parser.add_argument('-a', '--axis', default=True, action="store_false", help='Axis for cumulative counting (default=x axis)')
    parser.add_argument('-sh', '--show', default=True, action="store_false", help='Show output')
    args = parser.parse_args()

    device = depthai.Device('', False)

    config = {
        'streams': ['previewout', 'metaout'],
        'ai': {
            "blob_file": str(Path(args.model).resolve().absolute()),
            "blob_file_config": str(Path(args.config).resolve().absolute()),
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

    counter = [0, 0, 0, 0]  # left, right, up, down

    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackableObjects = {}

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

                height = frame.shape[0]
                width = frame.shape[1]

                objects = ct.update([
                    (int(detection.x_min * width), int(detection.y_min * height),
                    int(detection.x_max * width), int(detection.y_max * height))
                    for detection in detections
                ])

                for (objectID, centroid) in objects.items():
                    to = trackableObjects.get(objectID, None)

                    if to is None:
                        to = TrackableObject(objectID, centroid)
                    else:
                        if args.axis and not to.counted:
                            x = [c[0] for c in to.centroids]
                            direction = centroid[0] - np.mean(x)

                            if centroid[0] > args.roi_position*width and direction > 0:
                                counter[1] += 1
                                to.counted = True
                            elif centroid[0] < args.roi_position*width and direction < 0:
                                counter[0] += 1
                                to.counted = True
                            
                        elif not args.axis and not to.counted:
                            y = [c[1] for c in to.centroids]
                            direction = centroid[1] - np.mean(y)

                            if centroid[1] > args.roi_position*height and direction > 0:
                                counter[3] += 1
                                to.counted = True
                            elif centroid[1] < args.roi_position*height and direction < 0:
                                counter[2] += 1
                                to.counted = True

                        to.centroids.append(centroid)

                    trackableObjects[objectID] = to

                    text = "ID {}".format(objectID)
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)
                
                # Draw ROI line
                if args.axis:
                    cv2.line(frame, (int(args.roi_position*width), 0), (int(args.roi_position*width), height), (0xFF, 0, 0), 5)
                else:
                    cv2.line(frame, (0, int(args.roi_position*height)), (width, int(roi_position*height)), (0xFF, 0, 0), 5)

                # display count and status
                font = cv2.FONT_HERSHEY_SIMPLEX
                if args.axis:
                    cv2.putText(frame, f'Left: {counter[0]}; Right: {counter[1]}', (10, 35), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
                else:
                    cv2.putText(frame, f'Up: {counter[2]}; Down: {counter[3]}', (10, 35), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)

                if args.show:
                    cv2.imshow('cumulative_object_counting', frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break

    cap.release()
    cv2.destroyAllWindows()
    del device

if __name__ == '__main__':
    main()