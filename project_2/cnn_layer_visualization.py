"""
Created on Sat Nov 18 23:12:08 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np

import torch
from torch.optim import Adam
from torchvision import models

from misc_functions import preprocess_image, recreate_image, save_image
from PIL import Image

from ResNet import resnet34,resnet18,resnet50,resnet101,resnet152
import ResNet
from MyUtils import MyUtils


# import ResNet  

class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter):
        super().__init__()
        self.model = model
        # self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        self.myUtils=MyUtils()
        self.device=self.myUtils.deviceFun()
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        # self.hook_layer()
        # Generate a random image
        this_name = './pic2/'+ 'pic0, 4, 49, 53' +'.jpg'
                # 图片转成RGB，大坑!!!!!!!
        with Image.open(this_name).convert('RGB') as img:
            myImg=np.array(img)
        # Process image and return variable
        processed_image = preprocess_image(myImg, False)
        # processed_image = preprocess_image(random_image, False)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image.data
            # self.model=self.model.state_dict()
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image)
            # Save image
            if i % 5 == 0:
                im_path = '../generated/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)


if __name__ == '__main__':
    cnn_layer = 3
    filter_pos = 5
    # Fully connected layer is not needed
    # pretrained_model = models.vgg16(pretrained=False).features
    pretrained_model = torch.load(r'models/model.pth',map_location=torch.device('cuda:0'))
    pretrained_model=pretrained_model.features
    layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos)

    # Layer visualization with pytorch hooks
    layer_vis.visualise_layer_with_hooks()

    # Layer visualization without pytorch hooks
    # layer_vis.visualise_layer_without_hooks()
