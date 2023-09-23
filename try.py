import torch
import torchvision.transforms as T
import torch.nn.functional as fun
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from glob import glob
CUDA_LAUNCH_BLOCKING = 1
from networks.CDGNet import Res_Deeplab 
from utils.utils import decode_parsing, decode_parsing_agnostic, inv_preprocess

imgs = glob('/home/ubuntu/Vrushank/CDGNet/VITON-data/train/image/*')
print(len(imgs))

model = Res_Deeplab(22).cuda()
model.load_state_dict(torch.load('/home/ubuntu/Vrushank/CDGNet/snapshots/model_latest.pth'))
print('Done')
model.eval()

out_dir = '/home/ubuntu/Vrushank/CDGNet/VITON-data/train/image-parse-agnosticv3.2'
#out_dir1 = '/home/ubuntu/Vrushank/CDGNet/VITON-data/train/image-parse-agnostic'
#out_dir2 = '/home/ubuntu/Vrushank/CDGNet/VITON-data/parse-down'

if not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok = True)

#if not os.path.exists(out_dir1):
#    os.makedirs(out_dir1, exist_ok = True)

#if not os.path.exists(out_dir2):
#    os.makedirs(out_dir2, exist_ok = True)
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


transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_outputs(p):

    #og = cv2.imread(p)
    #og = cv2.resize(og, (768, 1024))
    name = p.split('/')[-1].split('.')[0]
    img = Image.open(p).convert('RGB')
    w, h = img.size # 
    img = transform(img)
    img = img.unsqueeze(0).cuda()

    with torch.no_grad():

        preds = model(img)

    #img_inv = inv_preprocess(img, 1)
    pred = fun.interpolate(preds[0][-1], (1024, 768), mode = 'bilinear')
    #p1 = pred.squeeze(0).cpu().numpy()
    #print(p1.shape)
    
    label = visualize_segmap(pred, tensor_out=False, agnostic=True)
    #print(label.getbands())
    #arr = np.array(label)
    #print(arr.max())
    #print(arr)
    label.save(f'{out_dir}/{name}.png')
    #y = label.cpu().numpy().transpose(2,1,0)
    #y = y * 255.0
    #print(type(y))
    #print(y.shape)
    #print(y.max())
    #cv2.imwrite('y.png', y)
    #y = Image.fromarray(y)

    
    #print(y.getbands())
    #y.save('y.png')
    #label = decode_parsing(pred, 1, is_pred  = True)
    #label_ag = decode_parsing_agnostic(pred, 1, is_pred  = True)

    #img1 = img_inv.squeeze(0).to(torch.uint8).cpu().numpy().transpose((1,2,0))
    #pred = label.squeeze(0).to(torch.uint8).cpu().numpy().transpose((1,2,0))
    #pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
    #pred = cv2.resize(pred, (w, h))
    #cv2.imwrite('x.png', pred)
    #cv2.imwrite(f'{out_dir}/{name}.png', pred)

    #pred1 = label_ag.squeeze(0).to(torch.uint8).cpu().numpy().transpose((1,2,0))
    #pred1 = cv2.cvtColor(pred1, cv2.COLOR_RGB2BGR)
    #pred1 = cv2.resize(pred1, (w, h))
    #cv2.imwrite(f'{out_dir1}/{name}.png', pred1)
    #pred_gs = np.argmax(pred1, axis = -1)
    #pred_gs = (pred_gs / 22) * 255
    #pred_gs = np.expand_dims(pred_gs, axis = -1)
    #pred_gs = cv2.resize(pred_gs, (768, 1024))
    #cv2.imwrite(f'{out_dir1}/{name}.png', pred_gs)
    #pred_gs_down = cv2.resize(pred_gs, (384, 512))
    #cv2.imwrite(f'{out_dir2}/{name}.png', pred_gs_down)
    #print(pred1.shape)
    #pred1 = cv2.cvtColor(pred1, cv2.COLOR_RGB2GRAY)
    #res = np.concatenate((og, pred1), axis = 1)
    
#for p in imgs:
#    get_outputs(p)
with ThreadPoolExecutor() as executor:

    executor.map(get_outputs, imgs)