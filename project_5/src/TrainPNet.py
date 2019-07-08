from MyTrain import MyTrain
if __name__ == "__main__":
    imgPath=r'/mnt/D/my_celebea/pic/12'
    tagPath=r'/mnt/D/my_celebea/txt/12list_bbox_celeba.txt'
    testTagPath=r'/mnt/D/my_celebea/txt/test12.txt'
    testImgPath=imgPath
    testResult=r'/mnt/D/my_celebea/txt/resultPNet.txt'
    myTrain=MyTrain(Net='PNet',epoch=100,batchSize=512,imgPath=imgPath,tagPath=tagPath,testTagPath=testTagPath,testImgPath=testImgPath,testResult=testResult)
    myTrain.train()