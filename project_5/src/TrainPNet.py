from MyTrain import MyTrain
if __name__ == "__main__":
    imgPath=r'/mnt/my_wider_face_train/12'
    tagPath=r'/mnt/my_wider_face_train/12list_wide_face.txt'

    testTagPath=r'/mnt/my_wider_face_val/12list_wide_face.txt'
    testImgPath=r'/mnt/my_wider_face_val/12'
    testResult=r'/mnt/my_wider_face_val/resultPNet.txt'

    myTrain=MyTrain(Net='PNet',epoch=10000,batchSize=1024,imgPath=imgPath,tagPath=tagPath,testTagPath=testTagPath,testImgPath=testImgPath,testResult=testResult)
    myTrain.train()