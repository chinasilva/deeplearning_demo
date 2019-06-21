from PIL import Image
import numpy as np

img=Image.open('./project_2/pic35.jpg')
# img.
im=np.array(img)

im[:,:,1]=0
im[:,:,2]=0
newImage=Image.fromarray(im)
newImage.show()