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
from utils import check_dir, set_random_seed, accuracy, instance_mIoU, get_logger
from models.second_segmentation import Segmentator
from data.transforms import get_transforms_binary_segmentation
from models.pretraining_backbone import ResNet18Backbone
from data.segmentation import DataReaderSemanticSegmentation
from utils.meters.averagevaluemeter import AverageValueMeter
import matplotlib.pyplot as plt
set_random_seed(0)
global_step = 0


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, help="folder containing the data")
    parser.add_argument('weights_init', type=str, default="ImageNet")
    parser.add_argument('--output-root', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--bs', type=int, default=8, help='batch_size')
    parser.add_argument('--size', type=int, default=256, help='image size')
    parser.add_argument('--snapshot-freq', type=int, default=1, help='how often to save models')
    parser.add_argument('--exp-suffix', type=str, default="", help="string to identify the experiment")
    args = parser.parse_args()

    hparam_keys = ["lr", "bs", "size"]
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(os.path.join(args.output_root, 'dt_binseg', args.exp_name))
    args.model_folder = check_dir(os.path.join(args.output_folder, "models"))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # Logging to the file and stdout
    logger = get_logger(args.output_folder, args.exp_name)
    img_size = (args.size, args.size)
    #args.data_folder = "segmentation_dataset\\COCO_mini5class_medium"
    args.data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data_folder)

    # model
    #args.weights_init = "results\\savedmodels\\binarysegmentation_best_model.pth"
    model = ResNet18Backbone(pretrained=False).cuda()
    pretrained_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.weights_init)
    pretrained = torch.load(pretrained_path)

    model = Segmentator(2, model.features, img_size).cuda()

    model.load_state_dict(pretrained['model'])
    model.decoder.last_conv = torch.nn.Sequential(*list(model.decoder.last_conv.children())[:-1],
                                                  torch.nn.Conv2d(256, 6, (1, 1)))
    model = model.cuda()


    # dataset
    train_trans, val_trans, train_target_trans, val_target_trans = get_transforms_binary_segmentation(args)
    data_root = args.data_folder
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

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),  lr=args.lr, momentum=0.9, weight_decay=1e-4)

    expdata = "  \n".join(["{} = {}".format(k, v) for k, v in vars(args).items()])
    logger.info(expdata)
    logger.info('train_data {}'.format(train_data.__len__()))
    logger.info('val_data {}'.format(val_data.__len__()))

    best_val_loss = np.inf
    best_val_miou = 0.0
    val_error_list = []
    val_miou_list = []
    train_error_list = []
    train_miou_list = []
    for epoch in range(25):
        logger.info("Epoch {}".format(epoch))
        print("================================================================")
        train_loss, train_miou = train(train_loader, model, criterion, optimizer, logger)
        val_results = validate(val_loader, model, criterion, logger, epoch)

        val_loss = val_results[0]
        val_miou = val_results[1]
        #val_loss_list = val_results[2]
        #val_miou_list = val_results[3]

        train_error_list.append(train_loss)
        train_miou_list.append(train_miou)
        val_error_list.append(val_loss)
        val_miou_list.append(val_miou)
        print('Validation miou and loss after epoch: ', epoch, " is ", val_miou, val_loss)

        # save model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if val_miou > best_val_miou:
            best_val_miou = val_miou

            save_model(model, optimizer, args, epoch, val_loss, val_miou, logger, best=True)
        else:
            save_model(model, optimizer, args, epoch, val_loss, val_miou, logger, best=False)

        print("Best validation loss: %f", best_val_loss)
        print("Best validation accuracy: %f ", best_val_miou)

        fig = plt.figure("Training error")
        plt.plot(np.arange(0, len(train_error_list)), train_error_list, label="Training loss")
        plt.xlabel("Epoch")
        plt.ylabel("Training error")
        plt.title("Training error captured after each epoch")
        fig.savefig("multi_segmentation_training_error")

        fig = plt.figure("Training miou")
        plt.plot(np.arange(0, len(train_miou_list)), train_miou_list, label="Training miou")
        plt.xlabel("Epoch")
        plt.ylabel("Training miou")
        plt.title("Mean Training miou captured after each epoch")
        fig.savefig("multi_segmentation_training_miou")

        fig = plt.figure("Mean validation error")
        plt.plot(np.arange(0, len(val_error_list)), val_error_list, label="validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Validation error")
        plt.title("Mean validation error captured after each epoch")
        fig.savefig("multi_segmentation_validation_error")

        fig = plt.figure("Mean validation miou")
        plt.plot(np.arange(0, len(val_miou_list)), val_miou_list, label="validation miou")
        plt.xlabel("Epoch")
        plt.ylabel("Validation miou")
        plt.title("Mean validation miou captured after each epoch")
        fig.savefig("multi_segmentation_validation_miou")


def train(loader, model, criterion, optimizer, logger):
    logger.info("Training")

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.train()

    loss_meter = AverageValueMeter()
    iou_meter = AverageValueMeter()
    time_meter = AverageValueMeter()
    steps_per_epoch = len(loader.dataset) / loader.batch_size

    start_time = time.time()
    batch_time = time.time()

    for idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        one = torch.sum(labels)
        labels = (labels * 255).long()
        labels = torch.squeeze(labels, dim=1)
        outputs = model(images)
        ones = torch.sum(labels)
        loss = criterion(outputs, labels)
        iou = instance_mIoU(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.add(loss.item())
        iou_meter.add(iou)
        time_meter.add(time.time() - batch_time)

        if idx % 50 == 0 or idx == len(loader)-1:
            text_print = "Epoch {:.4f} Avg loss = {:.4f} mIoU = {:.4f} Time {:.2f} (Total:{:.2f}) Progress {}/{}".format(
                        global_step / steps_per_epoch, loss_meter.mean, iou_meter.mean, time_meter.mean, time.time()-start_time, idx, int(steps_per_epoch))
            logger.info(text_print)


        batch_time = time.time()
    time_txt = "batch time: {:.2f} total time: {:.2f}".format(time_meter.mean, time.time() - start_time)
    logger.info(time_txt)
    return loss_meter.mean, iou_meter.mean

def validate(loader, model, criterion, logger, epoch=0):

    logger.info("Validating Epoch {}".format(epoch))
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()

    loss_meter = AverageValueMeter()
    iou_meter = AverageValueMeter()

    start_time = time.time()

    val_loss_list = []
    val_miou_list = []
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            one =  torch.sum(labels)
            labels = (labels * 255).long()
            ones = torch.sum(labels)
            labels = torch.squeeze(labels, dim=1)
            outputs = model(images)
            outputs = torch.nn.functional.interpolate(outputs.cuda(), size=labels.shape[-2:])

            loss = criterion(outputs, labels)
            iou = instance_mIoU(outputs, labels)

            loss_meter.add(loss.item())
            iou_meter.add(iou)

            if idx % 50 == 0:
                #val_miou_list.append(iou_meter.mean)
                #val_loss_list.append(loss_meter.mean)
                print('Validation accuracy and loss after image: ', idx, " is ", iou_meter.mean, loss_meter.mean)

    text_print = "Epoch {} Avg loss = {:.4f} mIoU = {:.4f} Time {:.2f}".format(epoch, loss_meter.mean, iou_meter.mean,
                                                                                   time.time() - start_time)
    logger.info(text_print)
    return loss_meter.mean, iou_meter.mean



def save_model(model, optimizer, args, epoch, val_loss, val_iou, logger, best=False):
    # save model
    add_text_best = 'BEST' if best else ''
    logger.info('==> Saving '+add_text_best+' ... epoch{} loss{:.03f} miou{:.03f} '.format(epoch, val_loss, val_iou))
    state = {
        'opt': args,
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': val_loss,
        'miou': val_iou
    }
    if best:
        torch.save(state, os.path.join(args.model_folder, 'ckpt_best.pth'))
    else:
        torch.save(state, os.path.join(args.model_folder, 'ckpt_epoch{}_loss{:.03f}_miou{:.03f}.pth'.format(epoch, val_loss, val_iou)))


if __name__ == '__main__':
    args = parse_arguments()
    print(vars(args))
    print()
    main(args)