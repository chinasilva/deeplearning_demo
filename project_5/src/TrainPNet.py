from MyTrain import MyTrain
if __name__ == "__main__":
    imgPath=r'/mnt/my_wider_face/12'
    tagPath=r'/mnt/my_wider_face/12list_wide_face.txt'
    testTagPath=r'/mnt/D/my_celebea/txt/test12.txt'
    testImgPath=imgPath
    testResult=r'/mnt/D/my_celebea/txt/resultPNet.txt'
    myTrain=MyTrain(Net='PNet',epoch=1000,batchSize=512,imgPath=imgPath,tagPath=tagPath,testTagPath=testTagPath,testImgPath=testImgPath,testResult=testResult)
    myTrain.train()