COCO_CLASS = ['person',
              "bicycle",
              "car",
              "bird",
              "cat",
              "dog",
              "horse",
              "sheep",
              "cow",
              "elephant",
              "bear",
              "zebra"
]
COCO_ANN_FILE = '/home/chinasilva/code/deeplearning_homework/project_6/data/instances_val2017.json'

LABEL_FILE = "/home/chinasilva/code/deeplearning_homework/project_6/data/coco_label.txt"
IMG_BASE_DIR = "/home/chinasilva/code/deeplearning_homework/project_6/data/coco100"

DEVICE = "cuda"

IMG_HEIGHT = 416
IMG_WIDTH = 416

CLASS_NUM = len(COCO_CLASS)

ANCHORS_GROUP = {
    13: [[116, 90], [156, 198], [373, 326]],
    26: [[30, 61], [62, 45], [59, 119]],
    52: [[10, 13], [16, 30], [33, 23]]
}

ANCHORS_GROUP_AREA = {
    13: [x * y for x, y in ANCHORS_GROUP[13]],
    26: [x * y for x, y in ANCHORS_GROUP[26]],
    52: [x * y for x, y in ANCHORS_GROUP[52]],
}

TAG_PATH="/home/chinasilva/code/deeplearning_homework/project_6/data/coco_label.txt"

IMG_PATH='/home/chinasilva/code/deeplearning_homework/project_6/data/val2017'

MODEL_PATH='/home/chinasilva/code/deeplearning_homework/project_6/model/model.pth'