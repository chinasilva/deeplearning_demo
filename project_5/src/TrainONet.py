from MyTrain import MyTrain

if __name__ == "__main__":
    imgPath=r'D:\my_celebea\pic\48'
    tagPath=r'D:\my_celebea\txt\48list_bbox_celeba.txt'
    myTrain=MyTrain(Net='ONet',epoch=100,batchSize=512,imgPath=imgPath,tagPath=tagPath)
    myTrain.train()