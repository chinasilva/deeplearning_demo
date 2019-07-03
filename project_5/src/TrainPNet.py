from MyTrain import MyTrain
if __name__ == "__main__":
    imgPath=r'D:\my_celebea\pic\12'
    tagPath=r'D:\my_celebea\txt\12list_bbox_celeba.txt'
    myTrain=MyTrain(Net='PNet',epoch=100,batchSize=512,imgPath=imgPath,tagPath=tagPath)
    myTrain.train()