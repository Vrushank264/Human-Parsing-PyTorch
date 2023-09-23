import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.utils as vutils
import torch.nn.functional as fun
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from tqdm import tqdm 
import argparse
import wandb

from networks.CDGNet import Res_Deeplab
from dataset.datasets import LIPDataSet, LIPDataValSet
from dataset.target_generation import generate_edge
from utils.utils import decode_parsing, inv_preprocess, AverageMeter
from utils.criterion import CriterionAll
from utils.miou import compute_mean_ioU
from evaluate import get_ccihp_pallete, valid


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """

    BATCH_SIZE = 8
    try:
        DATA_DIRECTORY = os.environ['SM_CHANNEL_TRAIN']
    except KeyError:
        DATA_DIRECTORY = '/home/vrushank/Spyne/HR-Viton/CCIHP'

    IGNORE_LABEL = 255
    INPUT_SIZE = '512, 512'
    LEARNING_RATE = 3e-4
    MOMENTUM = 0.9
    NUM_CLASSES = 22
    POWER = 0.9
    RANDOM_SEED = 1234
    try:
        RESTORE_FROM= 'resnet101-imagenet.pth'
    except FileNotFoundError:
        RESTORE_FROM = '/home/vrushank/Spyne/HR-Viton/CCIHP/resnet101-imagenet.pth'
    SAVE_NUM_IMAGES = 2
    SAVE_PRED_EVERY = 10000
    try:
        SNAPSHOT_DIR = '/opt/ml/checkpoints/'
    except KeyError:
        SNAPSHOT_DIR = 'snapshots/'

    WEIGHT_DECAY = 0.0005

    parser = argparse.ArgumentParser(description="CDG Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--dataset", type=str, default='train', choices=['train', 'val', 'trainval', 'test'],
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).") 
    parser.add_argument("--start-iters", type=int, default=0,
                        help="Number of classes to predict (including background).") 
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="choose the number of recurrence.")
    parser.add_argument("--num_epochs", type=int, default=150,
                        help="choose the number of recurrence.")
    
    return parser.parse_args()



def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, total_iters):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    args = get_arguments()
    lr = lr_poly(args.learning_rate, i_iter, total_iters, args.power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def set_bn_momentum(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003


def train(loader, valid_loader, model, opt, scaler, criterion, total_iters, epoch, args):


    model.train()
    loop = tqdm(loader, position = 0, leave = True)
    loss_ = AverageMeter()
    for idx, batch in enumerate(loop):

        idx += len(loader) * epoch
        lr = adjust_learning_rate(opt, idx, total_iters)

        imgs, labels, hgt, wgt, hwgt, _ = batch
        imgs, labels = imgs.cuda(non_blocking = True), labels.cuda(non_blocking = True)
        edges = generate_edge(labels)
        labels = labels.type(torch.cuda.LongTensor) #Check LongStorage which torch.cuda recommended
        edges = edges.type(torch.cuda.LongTensor)
        hgt = hgt.float().cuda(non_blocking = True)
        wgt = wgt.float().cuda(non_blocking = True)
        hwgt = hwgt.float().cuda(non_blocking = True)
        opt.zero_grad(set_to_none = True)

        with torch.cuda.amp.autocast_mode.autocast():

            preds = model(imgs)
            loss = criterion(preds, [labels, edges], [hgt, wgt, hwgt])
        
        loss_.update(loss.detach(), imgs.size(0))
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if idx % 500 == 0:

            wandb.log({'Training Loss': loss_.avg, 'Learning Rate': lr})
            print(f'Epoch [{epoch}/{args.num_epochs}] iter [{idx}/{len(loader)}] Learning Rate: {lr} Loss: {loss_.avg}')

        if idx % 2000 == 0:
            
            #print(imgs.shape)
            imgs_inv = inv_preprocess(imgs, args.save_num_images)
            labels_colors = decode_parsing(labels, args.save_num_images, is_pred = False)
            edges_colors = decode_parsing(edges, args.save_num_images, is_pred = False)
            #if isinstance(preds, list):
            #    preds = preds[0]
            pred = fun.interpolate(preds[0][-1],(512,512), mode='bilinear', align_corners=True )
            pred_edge = fun.interpolate(preds[1][-1],(512,512), mode='bilinear', align_corners=True )
            preds_colors = decode_parsing(pred, args.save_num_images, is_pred = True)
            #Check the position of edges in the list
            pred_edges_colors = decode_parsing(pred_edge, args.save_num_images, 2, is_pred = True)
            
            #preds_colors = fun.interpolate(preds_colors, (512, 512), mode = 'bilinear', align_corners = True)
            #pred_edges_colors = fun.interpolate(pred_edges_colors, (512, 512), mode = 'bilinear', align_corners = True)

            img = vutils.make_grid(imgs_inv*255, normalize = False, scale_each = True)
            lab = vutils.make_grid(labels_colors, normalize = False, scale_each = True)
            pred = vutils.make_grid(preds_colors, normalize = False, scale_each = True)
            edge = vutils.make_grid(edges_colors, normalize = False, scale_each = True)
            pred_edge = vutils.make_grid(pred_edges_colors, normalize = False, scale_each = True)

            img_wb = wandb.Image(img.to(torch.uint8).cpu().numpy().transpose((1,2,0)))
            labels_wb = wandb.Image(lab.to(torch.uint8).cpu().numpy().transpose((1,2,0)))
            pred_wb = wandb.Image(pred.to(torch.uint8).cpu().numpy().transpose((1,2,0)))
            edge_wb = wandb.Image(edge.to(torch.uint8).cpu().numpy().transpose((1,2,0)))
            pred_edge_wb = wandb.Image(pred_edge.to(torch.uint8).cpu().numpy().transpose((1,2,0)))

            wandb.log({
                'Images': img_wb,
                'Target': labels_wb,
                'Pred': pred_wb,
                'Edges': edge_wb,
                'Pred Edges': pred_edge_wb
            })
    
    if epoch % 2 == 0:

        num_samples = len(valid_loader) * args.batch_size
        parsing_preds, img, scales, centers = valid(model, valid_loader, [512, 512],  num_samples, 1)
        if isinstance(parsing_preds, np.ndarray):
            output_parsing = parsing_preds.copy()
        if isinstance(parsing_preds, torch.Tensor):
            output_parsing = parsing_preds.clone()
        else:
            output_parsing = parsing_preds

        mIoU, pixel_acc, mean_acc = compute_mean_ioU(parsing_preds, scales, centers, args.num_classes, args.data_dir, [512, 512])
        print('Printing MIoU Values...')
        for k, v in mIoU.items():
            print(f'{k}: {v}')
        print(f'Pixel Accuracy: {pixel_acc}')
        print(f'Mean Accuracy: {mean_acc}')
        palette = get_ccihp_pallete()
        wandb.log({
            'Valid MIoU': mIoU,
            'Valid Pixel Accuracy': pixel_acc,
            'Valid Mean Accuracy': mean_acc
            })
        print('Values Logged on wandb')
        for i in range(10):
            print('Inside Loop')
            #ip_img = Image.fromarray(img[i])
            op_img = Image.fromarray(output_parsing[i])
            op_img.putpalette(palette)
            #ip_img_wb = wandb.Image(ip_img)
            op_label_wb = wandb.Image(op_img)
            wandb.log({'Valid Pred': op_label_wb})

    


def main():

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    args = get_arguments()
    dataset = LIPDataSet(args.data_dir, args.dataset, [512, 512], transform = transform)
    train_loader = DataLoader(dataset, 
                            batch_size = args.batch_size, 
                            shuffle  = True,
                            num_workers = 4,
                            pin_memory = True)
    
    val_dataset = LIPDataValSet(args.data_dir, transform = transform)
    valid_loader = DataLoader(val_dataset,
                            batch_size = args.batch_size,
                            shuffle = False,
                            pin_memory= True)
    
    model = Res_Deeplab(num_classes = args.num_classes)
    print("Loading Model...")
    ckpt = torch.load(os.path.join(os.getcwd(), args.restore_from))
    new_params = model.state_dict().copy()

    for i in ckpt:
        i_parts = i.split('.')
        if not i_parts[0] == 'fc':
            new_params['.'.join(i_parts[0:])] = ckpt[i]
    
    model.load_state_dict(new_params)
    model.cuda()
    print('Model Loaded.')

    criterion = CriterionAll().cuda()
    opt = torch.optim.SGD(
        model.parameters(),
        lr = args.learning_rate,
        momentum = 0.9,
        weight_decay = args.weight_decay,
        nesterov = True
    )

    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    total_iters = len(train_loader) * args.num_epochs

    for epoch in range(args.num_epochs):

        train(train_loader, 
            valid_loader,
            model,
            opt,
            scaler,
            criterion,
            total_iters,
            epoch, 
            args
            )
    

if __name__ == '__main__':

    wandb.init(project = 'Human Parsing')
    main()


 