import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import random
import numpy as np
import argparse
import torch
import time
from torchvision.transforms import ToTensor
from utils.meters import AverageValueMeter
from utils.weights import load_from_weights
from utils import check_dir, set_random_seed, accuracy, mIoU, get_logger, save_in_log
from models.att_segmentation import AttSegmentator
from torch.utils.tensorboard import SummaryWriter
from data.transforms import get_transforms_binary_segmentation
from models.pretraining_backbone import ResNet18Backbone
from data.segmentation import DataReaderSingleClassSemanticSegmentationVector, DataReaderSemanticSegmentationVector
import matplotlib.pyplot as plt

set_random_seed(0)
global_step = 0

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, help="folder containing the data")
    parser.add_argument('--pretrained_model_path', type=str, default='')
    parser.add_argument('--output-root', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--bs', type=int, default=8, help='batch_size')
    parser.add_argument('--att', type=str, default='sdotprod', help='Type of attention. Choose from {additive, cosine, dotprod, sdotprod}')
    parser.add_argument('--size', type=int, default=256, help='image size')
    parser.add_argument('--snapshot-freq', type=int, default=5, help='how often to save models')
    parser.add_argument('--exp-suffix', type=str, default="")
    args = parser.parse_args()

    hparam_keys = ["lr", "bs", "att"]
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(os.path.join(args.output_root, 'dt_attseg', args.exp_name))
    args.model_folder = check_dir(os.path.join(args.output_folder, "models"))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # Logging to the file and stdout
    logger = get_logger(args.output_folder, args.exp_name)
    img_size = (args.size, args.size)

    #args.pretrained_model_path = "binary_segmentation.pth"
    # model
    pretrained_model = ResNet18Backbone(True).cuda()

    model = AttSegmentator(5, pretrained_model.features, att_type='additive').cuda()

    if os.path.isfile(args.pretrained_model_path):
        model = load_from_weights(model, args.pretrained_model_path, logger)

    # dataset
    #args.data_folder = "segmentation_dataset\\COCO_mini5class_medium"
    data_root = args.data_folder
    train_transform, val_transform, train_transform_mask, val_transform_mask = get_transforms_binary_segmentation(args)
    vec_transform = ToTensor()
    train_data = DataReaderSingleClassSemanticSegmentationVector(
        os.path.join(data_root, "imgs/train2014"),
        os.path.join(data_root, "aggregated_annotations_train_5classes.json"),
        transform=train_transform,
        vec_transform=vec_transform,
        target_transform=train_transform_mask
    )
    # Note that the dataloaders are different.
    # During validation we want to pass all the semantic classes for each image
    # to evaluate the performance.
    val_data = DataReaderSemanticSegmentationVector(
        os.path.join(data_root, "imgs/val2014"),
        os.path.join(data_root, "aggregated_annotations_val_5classes.json"),
        transform=val_transform,
        vec_transform=vec_transform,
        target_transform=val_transform_mask
    )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True,
                                               num_workers=6, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False,
                                             num_workers=6, pin_memory=True, drop_last=False)


    # TODO: loss
    criterion = torch.nn.CrossEntropyLoss().cuda()
    # TODO: SGD optimizer (see pretraining)
    optimizer = torch.optim.SGD(model.parameters(),  lr=args.lr, momentum=0.9, weight_decay=1e-4)

    expdata = "  \n".join(["{} = {}".format(k, v) for k, v in vars(args).items()])
    logger.info(expdata)
    logger.info('train_data {}'.format(train_data.__len__()))
    logger.info('val_data {}'.format(val_data.__len__()))

    best_val_loss = np.inf
    best_val_miou = 0.0

    train_loss_list = []
    train_miou_list = []
    val_loss_list = []
    val_miou_list =[]
    for epoch in range(20):
        logger.info("Epoch {}".format(epoch))
        train_loss, train_miou = train(train_loader, model, criterion, optimizer, logger)
        val_loss, val_miou = validate(val_loader, model, criterion, logger, epoch)
        train_loss_list.append(train_loss)
        train_miou_list.append(train_miou)
        val_loss_list.append(val_loss)
        val_miou_list.append(val_miou)

        # save model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if val_miou > best_val_miou:
            best_val_miou = val_miou

            model_save_dir = os.path.join(os.getcwd(), "results", "savedmodels")

            if not os.path.exists(model_save_dir):
                os.mkdir(model_save_dir)

            name = "attention_segmentation_model_new_"+str(epoch)+ ".pth"
            save_model_str = os.path.join(model_save_dir, name)
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': val_loss}, save_model_str)

        fig = plt.figure("Mean Training error")
        plt.plot(np.arange(0, len(train_loss_list)), train_loss_list, label="Training loss")
        plt.xlabel("Epoch")
        plt.ylabel("Training error")
        plt.title("Mean Training error captured after each epoch")
        fig.savefig("attention_segmentation_training_error")

        fig = plt.figure("Mean Training miou")
        plt.plot(np.arange(0, len(train_miou_list)), train_miou_list, label="Training miou")
        plt.xlabel("Epoch")
        plt.ylabel("Training miou")
        plt.title("Mean Training miou captured after each epoch")
        fig.savefig("attention_segmentation_training_miou")

        fig = plt.figure("Mean validation error")
        plt.plot(np.arange(0, len(val_loss_list)), val_loss_list, label="validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Validation error")
        plt.title("Mean validation error captured after each epoch")
        fig.savefig("attention_segmentation_validation_error")

        fig = plt.figure("Mean validation miou")
        plt.plot(np.arange(0, len(val_miou_list)), val_miou_list, label="validation miou")
        plt.xlabel("Epoch")
        plt.ylabel("Validation miou")
        plt.title("Mean validation miou captured after each epoch")
        fig.savefig("attention_segmentation_validation_miou")


def train(loader, model, criterion, optimizer, logger):
    logger.info("Training")
    model.train()

    loss_meter = AverageValueMeter()
    iou_meter = AverageValueMeter()
    time_meter = AverageValueMeter()
    steps_per_epoch = len(loader.dataset) / loader.batch_size

    start_time = time.time()
    batch_time = time.time()
    for idx, (img, v_class, label) in enumerate(loader):
        img = img.cuda()
        v_class = v_class.float().cuda().squeeze()
        logits, alphas = model(img, v_class, out_att=True)
        logits = logits.squeeze()
        labels = (torch.nn.functional.interpolate(label.cuda(), size=logits.shape[-2:]).squeeze(1) * 256).long()
        loss = criterion(logits, labels)
        iou = mIoU(logits, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.add(loss.item())
        iou_meter.add(iou)
        time_meter.add(time.time() - batch_time)

        if idx % 50 == 0 or idx == len(loader) - 1:
            text_print = "Epoch {:.4f} Avg loss = {:.4f} mIoU = {:.4f} Time {:.2f} (Total:{:.2f}) Progress {}/{}".format(
                global_step / steps_per_epoch, loss_meter.mean, iou_meter.mean, time_meter.mean,
                time.time() - start_time, idx, int(steps_per_epoch))
            logger.info(text_print)
            loss_meter.reset()
            iou_meter.reset()

        batch_time = time.time()
    time_txt = "batch time: {:.2f} total time: {:.2f}".format(time_meter.mean, time.time()-start_time)
    logger.info(time_txt)
    return loss_meter.mean, iou_meter.mean

def validate(loader, model, criterion, logger, epoch=0):
    logger.info("Validating Epoch {}".format(epoch))
    model.eval()

    loss_meter = AverageValueMeter()
    iou_meter = AverageValueMeter()

    start_time = time.time()
    for idx, (img, v_class, label) in enumerate(loader):
        img = img.squeeze(0).cuda()
        v_class = v_class.float().cuda().squeeze()
        logits, alphas = model(img, v_class, out_att=True)
        label = label.squeeze(0).unsqueeze(1)
        labels = (torch.nn.functional.interpolate(label.cuda(), size=logits.shape[-2:]).squeeze(1) * 256).long()
        loss = criterion(logits, labels)
        iou = mIoU(logits, labels)

        loss_meter.add(loss.item())
        iou_meter.add(iou)

        if idx % 50 == 0:
            print('Validation accuracy and loss after image: ', idx, " is ", iou_meter.mean, loss_meter.mean)

    text_print = "Epoch {} Avg loss = {:.4f} mIoU = {:.4f} Time {:.2f}".format(epoch, loss_meter.mean, iou_meter.mean, time.time()-start_time)
    logger.info(text_print)
    return loss_meter.mean, iou_meter.mean

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
