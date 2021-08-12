import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pprint
import random
import sys
sys.path.insert(0,os.getcwd())
import numpy as np
import argparse
import torch
import time
from utils import check_dir, set_random_seed, accuracy, mIoU, get_logger
from models.second_segmentation import Segmentator
from data.transforms import get_transforms_binary_segmentation
from models.pretraining_backbone import ResNet18Backbone
from data.segmentation import DataReaderBinarySegmentation, DataReaderSemanticSegmentation
from utils.meters.averagevaluemeter import AverageValueMeter
import matplotlib.pyplot as plt
set_random_seed(0)
global_step = 0

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--bs', type=int, default=2, help='batch_size')
    parser.add_argument('--size', type=int, default=256, help='image size')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_arguments()
    print(vars(args))
    print()

    '''
    error = [0.23843826016653277, 0.2153649703131104, 0.20655208169098024, 0.21074946282356896, 0.2072083375753753]
    
    fig = plt.figure("Mean validation error captured for every epoch")
    plt.plot(np.arange(0, 5), error, label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Validation error")
    plt.title("Mean validation error captured for every epoch")
    plt.legend()
    fig.savefig("self_supervised_validation_error_new")'''
    img_size = 256
    encoder_model = ResNet18Backbone(pretrained=False).cuda()

    """
    model = Segmentator(2, encoder_model.features, img_size).cuda()
    pretrained_path = "/home/kiran/kiran/dl_lab/week1/assignment/Kiran_Kumaraswamy_assignment1_new/Computer_Vision_Exercise_2021/results/dt_binseg/lr0.005_bs8_size256_/models/ckpt_epoch18_loss0.419_miou0.095.pth"
    pretrained = torch.load(pretrained_path)
    model.load_state_dict(pretrained['model'])
    # dataset
    print(pretrained["epoch"])
    train_trans, val_trans, train_target_trans, val_target_trans = get_transforms_binary_segmentation(args)
    data_root = "data/COCO_mini5class_medium/"
    train_data = DataReaderBinarySegmentation(
        os.path.join(data_root, "imgs/train2014"),
        os.path.join(data_root, "aggregated_annotations_train_5classes.json"),
        transform=train_trans,
        target_transform=train_target_trans
    )
    val_data = DataReaderBinarySegmentation(
        os.path.join(data_root, "imgs/val2014"),
        os.path.join(data_root, "aggregated_annotations_val_5classes.json"),
        transform=val_trans,
        target_transform=val_target_trans
    )
    print("Dataset size: {} samples".format(len(train_data)))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True,
                                               num_workers=6, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=6, pin_memory=True, drop_last=False)
    """ #Multi class plot
    pretrained_path = "/home/kiran/kiran/dl_lab/week1/assignment/Kiran_Kumaraswamy_assignment1_new/Computer_Vision_Exercise_2021/results/dt_binseg/lr0.005_bs8_size256_/models/ckpt_epoch18_loss0.419_miou0.095.pth"
    pretrained = torch.load(pretrained_path)

    model = Segmentator(6, encoder_model.features, img_size).cuda()

    model.load_state_dict(pretrained['model'])
    model = model.cuda()

    # dataset
    train_trans, val_trans, train_target_trans, val_target_trans = get_transforms_binary_segmentation(args)
    data_root = "data/COCO_mini5class_medium/"
    train_data = DataReaderSemanticSegmentation(
        os.path.join(data_root, "imgs/train2014"),
        os.path.join(data_root, "aggregated_annotations_train_5classes.json"),
        transform=train_trans,
        target_transform=train_target_trans
    )
    val_data = DataReaderSemanticSegmentation(
        os.path.join(data_root, "imgs/val2014"),
        os.path.join(data_root, "aggregated_annotations_val_5classes.json"),
        transform=val_trans,
        target_transform=val_target_trans
    )
    print("Dataset size: {} samples".format(len(train_data)))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True,
                                               num_workers=6, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False,
                                             num_workers=6, pin_memory=True, drop_last=False)


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()

    display_image = []
    sum = 0
    count = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            one = torch.sum(labels)
            labels = (labels * 255).long()
            ones = torch.sum(labels)
            labels = torch.squeeze(labels, dim=1)
            outputs = model(images)
            print(outputs.size())
            outputs = torch.nn.functional.interpolate(outputs.cuda(), size=labels.shape[-2:])
            miou = mIoU(outputs.float(), labels.float()).item()
            print(miou)
            sum+= miou
            if miou >= 0.5 and torch.sum(torch.argmax(outputs, dim=1) == 5) > 50.:
                count+=1
                print(idx)
                print(miou)
                display_image.append(images)
                outputs = outputs
                outputs = torch.nn.functional.interpolate(outputs.cuda(), size=labels.shape[-2:])
                outputs = torch.argmax(outputs, dim=1)
                display_image.append(labels)
                display_image.append(outputs)
                if count == 2:
                    break
        #print(sum/idx)

    fig, ax = plt.subplots(2, 3)
    img = torch.squeeze(display_image[0]).permute(1, 2, 0)
    ax[0][0].imshow(img.cpu())
    ax[0][0].set_title("Image")
    
    img = display_image[1].permute(1, 2, 0)
    ax[0][1].imshow(img.cpu())
    ax[0][1].set_title("Ground Truth")
    
    img = display_image[2].permute(1, 2, 0)
    ax[0][2].imshow(img.cpu())
    ax[0][2].set_title("Prediction")
    
    if count > 1:
        img = torch.squeeze(display_image[3]).permute(1, 2, 0)
        ax[1][0].imshow(img.cpu())
        ax[1][0].set_title("Image")
    
        img = display_image[4].permute(1, 2, 0)
        ax[1][1].imshow(img.cpu())
        ax[1][1].set_title("Ground Truth")
    
        img = display_image[5].permute(1, 2, 0)
        ax[1][2].imshow(img.cpu())
        ax[1][2].set_title("Prediction")
    
    plt.savefig("mt_display")
    '''
    fig = plt.figure(figsize=(64, 64))
    plt.title("Binary segmentation")
    img = torch.squeeze(display_image[0]).permute(1, 2, 0)
    fig.add_subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(img.cpu())
    img = display_image[1].permute(1, 2, 0)
    fig.add_subplot(2, 3, 2)
    plt.title("Ground truth")
    plt.imshow(img.cpu())

    img = display_image[2].permute(1, 2, 0)
    fig.add_subplot(2, 3, 3)
    plt.title("Prediction")
    plt.imshow(img.cpu())


    if count > 1:
        img = torch.squeeze(display_image[3]).permute(1, 2, 0)
        fig.add_subplot(2, 3, 4)
        plt.imshow(img.cpu())
        img = display_image[4].permute(1, 2, 0)
        fig.add_subplot(2, 3, 5)
        plt.imshow(img.cpu())

        img = display_image[5].permute(1, 2, 0)
        fig.add_subplot(2, 3, 6)
        plt.imshow(img.cpu())

    plt.savefig("bt1_display")'''

