import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import argparse
import torch
from pprint import pprint
from data.pretraining import DataReaderPlainImg, custom_collate
from data.transforms import get_transforms_pretraining
from utils import check_dir, accuracy, get_logger
from models.pretraining_backbone import ResNet18Backbone
from utils.meters.averagevaluemeter import AverageValueMeter
import matplotlib.pyplot as plt
from utils import check_dir, set_random_seed, accuracy, mIoU, get_logger
from tqdm.autonotebook import tqdm
global_step = 0


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, help="folder containing the data (crops)",
                        default='data/crops/images/256')
    parser.add_argument('--weights-init', type=str,
                        default="data/pretrain_weights_init.pth")
    parser.add_argument('--output-root', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--bs', type=int, default=4, help='batch_size')
    parser.add_argument("--size", type=int, default=256, help="size of the images to feed the network")
    parser.add_argument('--snapshot-freq', type=int, default=1, help='how often to save mSodels')
    parser.add_argument('--exp-suffix', type=str, default="", help="string to identify the experiment")

    args = parser.parse_args()

    hparam_keys = ["lr", "bs", "size"]
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(os.path.join(args.output_root, 'pretrain', args.exp_name))
    args.model_folder = check_dir(os.path.join(args.output_folder, "models"))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # Logging to the file and stdout
    logger = get_logger(args.output_folder, args.exp_name)

    # build model and load weights
    model = ResNet18Backbone(pretrained=False).cuda()

    #args.weights_init = "results/savedmodels/pretrain_weights_init.pth"
    pretrain_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 args.weights_init)

    pretrained = torch.load(str(pretrain_path))
    model.load_state_dict(pretrained['model'])

    # load dataset
    #args.data_folder = 'data/crops/images'
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             args.data_folder)

    args.data_folder = data_path
    data_root = args.data_folder
    train_transform, val_transform = get_transforms_pretraining(args)
    train_data = DataReaderPlainImg(os.path.join(data_root, str(args.size), "train"), transform=train_transform)
    val_data = DataReaderPlainImg(os.path.join(data_root, str(args.size), "val"), transform=val_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=2,
                                               pin_memory=True, drop_last=True, collate_fn=custom_collate)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=2,
                                             pin_memory=True, drop_last=True, collate_fn=custom_collate)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    optimizer.load_state_dict(pretrained['optimizer'])

    expdata = "  \n".join(["{} = {}".format(k, v) for k, v in vars(args).items()])
    logger.info(expdata)
    logger.info('train_data {}'.format(train_data.__len__()))
    logger.info('val_data {}'.format(val_data.__len__()))

    best_val_loss = np.inf
    best_val_accuracy = 0

    train_accuracy_list = []
    train_loss_list = []
    val_loss_list = []
    val_accuracy_list = []
    # Train-validate for one epoch. You don't have to run it for 100 epochs, preferably until it starts overfitting.
    for epoch in range(10):
        print("Epoch {}".format(epoch))
        train_accuracy, train_loss = train(train_loader, model, criterion, optimizer)
        val_loss, val_acc = validate(val_loader, model, criterion)
        print('Validation accuracy and loss after epoch: ', epoch, " is ", val_acc, val_loss)

        train_accuracy_list.append(train_accuracy)
        train_loss_list.append(train_loss)

        val_accuracy_list.append(val_acc)
        val_loss_list.append(val_loss)
        # save model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc

            model_save_dir = os.path.join(os.getcwd(), "results", "savedmodels")

            if not os.path.exists(model_save_dir):
                os.mkdir(model_save_dir)

            name = "pretrain_best_model_new_"+str(epoch)+ ".pth"
            save_model_str = os.path.join(model_save_dir, name)
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': val_loss}, save_model_str)

        print("Best validation loss: ", best_val_loss)
        print("Best validation accuracy: ", best_val_accuracy)

        fig = plt.figure("Mean Training error captured for every epoch")
        plt.plot(np.arange(0, len(train_loss_list)), train_loss_list, label="Training loss")
        plt.xlabel("Epoch")
        plt.ylabel("Training error")
        plt.title("Mean Training error captured for every epoch")
        fig.savefig("self_supervised_training_error_new")

        fig = plt.figure("Mean Training accuracy captured for every epoch")
        plt.plot(np.arange(0, len(train_accuracy_list)), train_accuracy_list, label="Training miou")
        plt.xlabel("Epoch")
        plt.ylabel("Training accuracy")
        plt.title("Mean Training accuracy captured captured for every epoch")
        fig.savefig("self_supervised_training_accuracy_new")

        fig = plt.figure("Mean validation error captured for every epoch")
        plt.plot(np.arange(0, len(val_loss_list)), val_loss_list, label="validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Validation error")
        plt.title("Mean validation error captured for every epoch")
        fig.savefig("self_supervised_validation_error_new")

        fig = plt.figure("Mean validation accuracy captured for every epoch")
        plt.plot(np.arange(0, len(val_accuracy_list)), val_accuracy_list, label="validation miou")
        plt.xlabel("Epoch")
        plt.ylabel("Validation accuracy")
        plt.title("Mean validation accuracy captured for every epoch")
        fig.savefig("self_supervised_validation_accuracy_new")


# train one epoch over the whole training dataset. You can change the method's signature.
def train(loader, model, criterion, optimizer):
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    score = AverageValueMeter()
    losses = AverageValueMeter()

    model.train()
    train_accuracy = []
    train_loss = []
    loader = tqdm(loader)
    for idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predict = model(images)

        loss = criterion(predict, labels)
        loss.backward()
        optimizer.step()

        acc = accuracy(predict, labels)

        n = images.size(0)
        losses.add(loss.item())
        score.add(acc[0])

        if idx != 0 and idx % 100 == 0:
            print('Training accuracy and loss after batch: ', idx, " is ", score.mean, losses.mean)

    return score.mean, losses.mean


# validation function. you can change the method's signature.
def validate(loader, model, criterion):
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()
    val_loss = AverageValueMeter()
    val_accuracy = AverageValueMeter()

    val_loss_list = []
    val_acc_list = []

    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device).long()

            outputs = model(images)

            loss = criterion(outputs, labels)

            acc = accuracy(outputs, labels)
            val_accuracy.add(acc[0])
            val_loss.add(loss.item())

            if idx != 0 and idx % 500 == 0:
                val_acc_list.append(val_accuracy.mean)
                val_loss_list.append(val_loss.mean)
                print('Validation accuracy and loss after image: ', idx, " is ", val_accuracy.mean, val_loss.mean)

    return val_loss.mean, val_accuracy.mean


if __name__ == '__main__':
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args)
