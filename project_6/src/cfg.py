COCO_ANN_FILE = '/home/chinasilva/code/deeplearning_homework/project_6/data/instances_train2017.json'

LABEL_FILE = "/home/chinasilva/code/deeplearning_homework/project_6/data/coco_label.txt"
IMG_BASE_DIR = "/home/chinasilva/code/deeplearning_homework/project_6/data/coco100"

DEVICE = "cuda"

IMG_HEIGHT = 416
IMG_WIDTH = 416

# COCO_CLASS = ['person',
#               "car",
#               "motorcycle",
#               "airplane",
#               "train",
#               "cow",
#               "elephant",
#               "bear",
#               "zebra",
#               "sandwich"
#               ]

COCO_CLASS = ['person',
              "bicycle",
              "car",
              "motorcycle",
              "airplane",
              "bus",
              "train",
              "truck",
              "boat",
              "traffic light",
              "fire hydrant",
              "stop sign",
              "parking meter",
              "bench",
              "bird",
              "cat",
              "dog",
              "horse",
              "sheep",
              "cow",
              "elephant",
              "bear",
              "zebra",
              "giraffe",
              "backpack",
              "umbrella",
              "handbag",
              "tie",
              "suitcase",
              "frisbee",
              "skis",
              "snowboard",
              "sports ball",
              "kite",
              "baseball bat",
              "baseball glove",
              "skateboard",
              "surfboard",
              "tennis racket",
              "bottle",
              "wine glass",
              "cup",
              "fork",
              "knife",
              "spoon",
              "bowl",
              "banana",
              "apple",
              "sandwich",
              "orange",
              "broccoli",
              "carrot",
              "hot dog",
              "pizza",
              "donut",
              "cake",
              "chair",
              "couch",
              "potted plant",
              "bed",
              "dining table",
              "toilet",
              "tv",
              "laptop",
              "mouse",
              "remote",
              "keyboard",
              "cell phone",
              "microwave",
              "oven",
              "toaster",
              "sink",
              "refrigerator",
              "book",
              "clock",
              "vase",
              "scissors",
              "teddy bear",
              "hair drier",
              "toothbrush"
]
# IMAGES_ID=[
# 1,16,19,57,63,69,80,90,108,128,155,171,178,180,183,191,202,212,251
# ]
# COCO_CLASS =['cat', 'dog', 'snowboar']
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

TAG_PATH="/home/chinasilva/code/deeplearning_homework/project_6/data/person_label.txt"

IMG_PATH='/home/chinasilva/code/deeplearning_homework/project_6/data'