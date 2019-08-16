import torch
def device_fun():
    device=torch.device("cuda:0" if torch.cuda.is_available() else"cpu")
    return device