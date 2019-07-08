from MyTrain import MyTrain

if __name__ == "__main__":
    imgPath=r'/mnt/D/my_celebea/pic/24'
    tagPath=r'/mnt/D/my_celebea/txt/24list_bbox_celeba.txt'
    testTagPath=r'/mnt/D/my_celebea/txt/test24.txt'
    testImgPath=imgPath
    testResult=r'/mnt/D/my_celebea/txt/resultRNet.txt'
    myTrain=MyTrain(Net='RNet',epoch=100,batchSize=512,imgPath=imgPath,tagPath=tagPath,testTagPath=testTagPath,testImgPath=testImgPath,testResult=testResult)
    myTrain.train()