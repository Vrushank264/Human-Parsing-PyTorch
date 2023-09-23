import torch
import torch.nn as nn
import torch.nn.functional as fun
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import os

from networks.CDGNet import Res_Deeplab

net = Res_Deeplab(22).cuda()
net.load_state_dict(torch.load(''))

data = cv2.imread()
data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
data = cv2.resize(data, (512,512))
data = torch.from_numpy(data[None]).to('cuda')

def visualize_segmap(input, multi_channel=True, tensor_out=True, batch=0, agnostic = False) :
    
    if not agnostic:
        palette = [
            0, 0, 0, 128, 0, 0, 254, 0, 0, 0, 85, 0, 169, 0, 51,
            254, 85, 0, 0, 0, 85, 0, 119, 220, 85, 85, 0, 0, 85, 85,
            85, 51, 0, 52, 86, 128, 0, 128, 0, 0, 0, 254, 51, 169, 220,
            0, 254, 254, 85, 254, 169, 169, 254, 85, 254, 254, 0, 254, 169, 0,
            0,0,0,0,0,0,0,0,0
        ]
    if agnostic:
        palette = [
            0, 0, 0, 128, 0, 0, 254, 0, 0, 0, 0, 0, 169, 0, 51,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 85, 0, 0, 85, 85,
            0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 254, 0, 0, 0,
            0, 0, 0, 85, 254, 169, 169, 254, 85, 254, 254, 0, 254, 169, 0,
            0,0,0,0,0,0,0,0,0
        ]
    input = input.detach()
    if multi_channel :
        input = ndim_tensor2im(input,batch=batch)
    else :
        input = input[batch][0].cpu()
        input = np.asarray(input)
        input = input.astype(np.uint8)
    input = Image.fromarray(input, 'P')
    input.putpalette(palette)

    if tensor_out :
        trans = T.ToTensor()
        return trans(input.convert('RGB'))

    return input


def ndim_tensor2im(image_tensor, imtype=np.uint8, batch=0):
    image_numpy = image_tensor[batch].cpu().float().numpy()
    result = np.argmax(image_numpy, axis=0)
    return result.astype(imtype)



class WrappedModel(nn.Module):

    def __init__(self, model):

        super().__init__()
        self.model = model
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().reshape([1, 3, 1, 1])
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda().reshape([1, 3, 1, 1])

    @torch.inference_mode()
    def forward(self, data, fp16=True):

        data = data.permute(0, 3, 1, 2).contiguous()
        data = data.div(255).sub(self.mean).div_(self.std)
        pred = self.model(data)
        pred = fun.interpolate(pred[0][-1], (1024, 768), mode = 'bilinear')
        
        return pred.contiguous()
    

wrp_model = WrappedModel(net).cuda().eval()
torch.cuda.synchronize()

with torch.no_grad():
    svd_out = wrp_model(data)

torch.cuda.synchronize()
print(svd_out.shape)

w1 = visualize_segmap(svd_out, tensor_out = False)
w1.save('w1.png')

OUT_PATH = "out"
os.makedirs(OUT_PATH, exist_ok=True)

wrp_model = wrp_model.half()

with torch.inference_mode(), torch.jit.optimized_execution(True):
    traced_script_module = torch.jit.trace(wrp_model, data)
    traced_script_module = torch.jit.optimize_for_inference(
        traced_script_module)


print(traced_script_module.code)
print(f"{OUT_PATH}/model.pt")
traced_script_module.save(f"{OUT_PATH}/model.pt")

traced_script_module = torch.jit.load(f"{OUT_PATH}/model.pt")

torch.cuda.synchronize()
with torch.no_grad():
    o = traced_script_module(data)
torch.cuda.synchronize()
print(o.shape)
w2 = visualize_segmap(o, tensor_out = False)
w2.save('w2.png')



