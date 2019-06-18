import torch
import cv2
import numpy as np

class MyUtils():
    def __init__(self):
        return
    def make_one_hot(self,labels, C=2):
        '''
        Converts an integer label torch.autograd.Variable to a one-hot Variable.
        
        Parameters
        ----------
        labels : torch.autograd.Variable of torch.cuda.LongTensor
            N x 1 x H x W, where N is batch size. 
            Each value is an integer representing correct classification.
        C : integer. 
            number of classes in labels.
        
        Returns
        -------
        target : torch.autograd.Variable of torch.cuda.FloatTensor
            N x C x H x W, where C is class number. One-hot encoded.
        '''

        labels=labels.long()
        one_hot= torch.zeros(labels.size()[0], C)
        target=one_hot.scatter_(1, labels.view(-1,1), 1)
        # print("labels:",labels)
        # print("labels.view:",labels.view(-1,1))
        # target = Variable(target)
            
        return target

    def deviceFun(self):
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        return device

    def changeChannel(self):
        input_path = '.\project_2\logo.jpg'
        output_path = '.\project_2\logo4channel.jpg'
        
        img = cv2.imread(input_path)

        b_channel, g_channel, r_channel = cv2.split(img)

        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
        # 最小值为0
        alpha_channel[:, :int(b_channel.shape[0] / 2)] = 0

        img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

        cv2.imwrite(output_path,img_BGRA)


if __name__ == "__main__":
    utils=MyUtils()
    utils.changeChannel()



