from MyTrain import MyTrain

if __name__ == "__main__":
    imgPath=r'/mnt/D/my_celebea/pic/48'
    tagPath=r'/mnt/D/my_celebea/txt/48list_bbox_celeba.txt'
    testTagPath=r'/mnt/D/my_celebea/txt/test48.txt'
    testImgPath=imgPath
    testResult=r'/mnt/D/my_celebea/txt/resultONet.txt'
    myTrain=MyTrain(Net='ONet',epoch=100,batchSize=512,imgPath=imgPath,tagPath=tagPath,testTagPath=testTagPath,testImgPath=testImgPath,testResult=testResult)
    myTrain.train()