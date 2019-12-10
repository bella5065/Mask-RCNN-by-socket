import cv2
import numpy as np
import os
import sys
from samples import coco
from mrcnn import utils
from mrcnn import model as modellib
import matplotlib.pyplot as plt
import socket
import pickle
import struct
import datetime

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

HOST="116.89.189.30"
PORT=9091

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST,PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)
model.load_weights(COCO_MODEL_PATH, by_name=True)
class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]


def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


colors = random_colors(len(class_names))
class_dict = {
    name: color for name, color in zip(class_names, colors)
}


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image

#Blur
def get_matrix_mask(image_mask, mask, label):

    #set in the matrix the value of the object label

    positions_mask = np.where(mask == 1)
    for i, j in zip(positions_mask[0], positions_mask[1]):
        image_mask[i][j] = label
    return image_mask

def blur_image(image, masked_image):
    
    blurred_image = image.astype(np.unit8).copy()
    blurred_image = cv2.blur(blurred_image, (10,10))
    positions_background = np.where(masked_image == 0)

    for i, j in zip(positions_background[0], positions_background[1]):
        blurred_image[i][j] = image[i][j]

    return blurred_image

def display_blurred(image, boxes, masks, ids, names, scores):
    N = boxes.shape[0]

    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    masked_image = np.zeros(shape=image.shape)

    for i in range(N):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]

        if ids[i] == 1:
            label = names[ids[i]]
            color = class_dict[label]
            score = scores[i] if scores is not None else None
            caption = '{} {:.2f}'.format(label, score) if score else label

            mask = masks[:, :, i]
            masked_image = get_matrix_mask(masked_image, mask, ids[i] + 1)

    blurred_image = blur_image(image, masked_image)

    return blurred_image


def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]

        if ids[i] == 1:
            label = names[ids[i]]
            color = class_dict[label]
            score = scores[i] if scores is not None else None
            caption = '{} {:.2f}'.format(label, score) if score else label
            mask = masks[:, :, i]

            image = apply_mask(image, mask, color)
        #image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        #image = cv2.putText(
        #    image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        #)

    return image


if __name__ == '__main__':
    """
        test everything
    """
    #Modified by Syeong#
    #input_video = sys.argv[1]
    #capture = cv2.VideoCapture(input_video)

    # these 2 lines can be removed if you dont have a 1080p camera.
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Recording Video
    conn,addr=s.accept()

    fcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(datetime.datetime.now().strftime("%d_%H-%M-%S.avi"), fcc, 5.0, (320,240))

    data = b""
    payload_size = struct.calcsize(">L")
    print("payload_size: {}".format(payload_size))

    while True:
        while len(data) < payload_size:
            print("Recv: {}".format(len(data)))
            data += conn.recv(1024)

        print("Done Recv: {}".format(len(data)))
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        print("msg_size: {}".format(msg_size))
        while len(data) < msg_size:
            data += conn.recv(1024)
        frame_data = data[:msg_size]
        data = data[msg_size:]

        frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        frame = cv2.flip(frame, 0)

        results = model.detect([frame], verbose=0)
        r = results[0]
        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
        )
        cv2.imshow('frame', frame)

        # Recording Video
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

 
    cv2.destroyAllWindows()
    out.release()
    s.close()
#Modified by Syeong#
