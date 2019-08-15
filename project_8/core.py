import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from src.main import device_fun

device=device_fun() 


def core(net, img_path):

    net.eval()

    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.CenterCrop(28),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    img = transform(Image.open(img_path)).unsqueeze(0) * -1
    # print("img:{}".format(img))
    img = img.to(device)
    output = net(img).to(device)

    
    # generate probability
    ouptut_prob = {}
    print("output:{}".format(output))

    for i, value in enumerate(F.softmax(output, dim=1)[0]):
        print("{}:{}".format(i,value))
        ouptut_prob[i] = str(round(value.data.item() * 100, 2)) + "%"
    return {
        "max_number": str(torch.argmax(output).cpu().numpy()),
        # "max_number": str(torch.argmax(output).numpy()),
        "probability": ouptut_prob
    }

if __name__ == "__main__":
    net = torch.jit.load("models/net.pth")
    result_data = core(net, "images/3.jpg")
    print(result_data)

