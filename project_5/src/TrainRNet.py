from MyTrain import MyTrain

if __name__ == "__main__":
    imgPath=r'/mnt/D/my_celebea/pic/24'
    tagPath=r'/mnt/D/my_celebea/txt/24list_bbox_celeba.txt'
    myTrain=MyTrain(Net='RNet',epoch=100,batchSize=512,imgPath=imgPath,tagPath=tagPath)
    myTrain.train()