from pycocotools.coco import COCO
from cfg import *

with open(LABEL_FILE, "w+") as f:
    coco = COCO(COCO_ANN_FILE)
    catIds = coco.getCatIds(catNms=COCO_CLASS)

    cats = {}
    for cat in coco.loadCats(catIds):
        cats[cat['id']] = cat['name']

    imgIds = coco.getImgIds(catIds=catIds)

    for imgId in imgIds:

        img = coco.loadImgs(imgId)[0]

        img_file_name = img['file_name']
        w, h = img['height'], img['width']
        w_scale, h_scale = w / IMG_WIDTH, h / IMG_HEIGHT

        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        f.write(img_file_name)
        for ann in anns:
            category_name = cats[ann['category_id']]
            cls = COCO_CLASS.index(category_name)  # 转换为我们的分类索引

            _x1, _y1, _w, _h = ann['bbox']
            _w0_5, _h0_5 = _w / 2, _h / 2
            _cx, _cy = _x1 + _w0_5, _y1 + _h0_5
            x1, y1, w, h = int(_cx / w_scale), int(_cy / h_scale), int(_w / w_scale), int(_h / h_scale)
            f.write(" {} {} {} {} {}".format(cls, x1, y1, w, h))
        f.write("\n")
        f.flush()
