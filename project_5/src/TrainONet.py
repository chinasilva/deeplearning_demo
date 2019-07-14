from MyTrain import MyTrain

if __name__ == "__main__":
    imgPath=r'/mnt/my_wider_face/48'
    tagPath=r'/mnt/my_wider_face/48list_wide_face.txt'
                                
    testTagPath=r'/mnt/my_wider_face_val/48list_wide_face.txt'
    testImgPath=r'/mnt/my_wider_face_val/48'
    testResult=r'/mnt/my_wider_face_val/resultONet.txt'
    myTrain=MyTrain(Net='ONet',epoch=1000,batchSize=512,imgPath=imgPath,tagPath=tagPath,testTagPath=testTagPath,testImgPath=testImgPath,testResult=testResult)
    myTrain.train()