from PIL import Image,ImageFont,ImageDraw
import random

savePath=r'/home/chinasilva/code/deeplearning_homework/project_9/data/'
w=200
h=80
# 自定义背景色
def randBackgroundColor():
    return (random.randint(0,160),random.randint(0,160),random.randint(0,160))

# 自定义文字颜色
def randFontColor():
    return (random.randint(140,255),random.randint(140,255),random.randint(140,255))

def createImg():
    img=Image.new(mode='RGB',size=(w,h))

#     img.show()
    return img

def loadFont():
#     print(ImageFont.load_default())
    return ImageFont.truetype('/usr/share/fonts/truetype/Sarai/Sarai.ttf',size=30)

# def randDigital():
#     return random.randint(0,9)

def randABC():
    str=""
    for i in range(26):
        str+=(chr(97+i))
    for j in range(10):
        str+=(chr(48+j))
    return random.choice(str)

def start():
    for i in range(1000):
        print("--------------------------",str(i))
        image=createImg()
        draw=ImageDraw.Draw(image)
        for x in range(w):
            for y in range(h):
                draw.point((x, y), fill=randBackgroundColor())
        values=""
        for i in range(4):
            value=randABC()
            draw.text((40+40*i,20),value,fill=randFontColor(), font=loadFont())
            values+=value
        image.save(savePath+values+".png","png")

if __name__ == "__main__":
    start()