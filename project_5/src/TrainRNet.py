from MyTrain import MyTrain

if __name__ == "__main__":
    imgPath=r'/mnt/my_wider_face_train/24'
    tagPath=r'/mnt/my_wider_face_train/24list_wide_face.txt'

    testTagPath=r'/mnt/my_wider_face_val/24list_wide_face.txt'
    testImgPath=r'/mnt/my_wider_face_val/24'
    testResult=r'/mnt/my_wider_face_val/resultRNet.txt'

    myTrain=MyTrain(Net='RNet',epoch=1000,batchSize=512,imgPath=imgPath,tagPath=tagPath,testTagPath=testTagPath,testImgPath=testImgPath,testResult=testResult)
    myTrain.train()