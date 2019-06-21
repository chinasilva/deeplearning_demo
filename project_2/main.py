from MyTrain import MyTrain


if __name__ == "__main__":
    path=r"./pic2"
    epoch=500
    batchSize=300
    myTrain=MyTrain(path,epoch,batchSize)
    myTrain.train()