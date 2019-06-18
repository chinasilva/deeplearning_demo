'''
训练模型前先使用此方法进行处理图片，
使用文件名标注坐标
'''
from PIL import Image
import os,re
import numpy as np
_RAW_DIR = "./pic"
_OUT_DIR="./pic2"
_RE_INDEX = re.compile(u"pic(\d*)\..+")



def parse_start():
    """
    parse the starter index in `./raw` dir
    """
    starter = os.listdir("./pic")[0]
    res = _RE_INDEX.match(starter)
    if not res:
        raise ValueError("No Files Found!")
    else:
        return int(res.group(1))

def resize_small(img):
    resized = img.resize((100,100), resample=Image.BICUBIC)
    return resized

def main():
    starter = parse_start()
    file_cound = len(os.listdir(_RAW_DIR))
    log_pic = '.\project_2\logo4channel.jpg'
    with Image.open(log_pic) as log_img:
        for index in range(starter, starter + file_cound):
            try:
                this_name = _RAW_DIR+'/pic'+ str(index) +'.jpg'
                # 图片转成RGB，大坑!!!!!!!
                with Image.open(this_name).convert('RGB') as img:
                     #批量更改尺寸
                    # _small = resize_small(img)
                    # _small.save(_RAW_DIR+'/pic'+ str(index) +'.jpg', format="jpeg")

                    # 随机logo尺寸
                    log_size=np.random.randint(30,50)
                    x_size=np.random.randint(0,100-log_size)
                    y_size=np.random.randint(0,100-log_size)


                    x2=x_size+log_size
                    y2=y_size+log_size
                    box = (x_size, y_size, x2, y2)

                    log_img2=log_img.resize((log_size,log_size))
                    img.paste(log_img2,box)
                    img.save(_OUT_DIR+'/pic'+ str(box).replace('(','').replace(')','') +'.jpg', format="jpeg")
            except:
                print(_RAW_DIR+'/pic'+ str(index) +'.jpg')
 

if __name__ == "__main__":
    main()

# self.myData=MyData(self.path)
# self.trainData=data.DataLoader(self.myData,batch_size=self.batchSize,shuffle=True)
