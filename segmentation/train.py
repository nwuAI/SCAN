
import torch
import argparse
import os
import numpy as np
from Modules.mriV import LoadMRIData
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils.lr_scheduler import LR_Scheduler
import Modules.lossmouleV as lm
# from torch.nn import DataParallel
# import nibabel as nib
from Modules import preModel
import config
from tensorboardX import SummaryWriter

SAVE_DIR ='./save_model/malc/train'  # directory you want to save the model
NUM_CLASS=28 # for MALC-coarse grained use 28/ for dhcp=10 /for pwml=5
RESUME_PATH = None  # directory you want to resume the model
RESUME_PRETRAIN =False

DATA_DIR ='./datasets/MALC2012'
DATA_LIST ='./data_list'

class Solver():
    def __init__(self, args, num_class):
        self.args = args
        self.num_class = num_class

        # 导入数据
        train_data = LoadMRIData(args.data_dir, args.data_list, 'train', num_class, num_slices=args.num_slices,
                                 se_loss=args.se_loss, use_weight=args.use_weights, Encode3D=args.encode3D)
        self.train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                       pin_memory=True)

        test_data = LoadMRIData(args.data_dir, args.data_list, 'test', num_class, num_slices=args.num_slices,
                                se_loss=False, Encode3D=args.encode3D)
        self.test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

        # 模型 #
        configs = config.get_biformer_configs()
        model = preModel.preBiformer(configs,num_class,args.num_slices)

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                        momentum=0.9, weight_decay=args.weight_decay)

        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))
        self.criterion = lm.CombinedLoss()
        self.model, self.optimizer = model, optimizer

        # Using cuda
        if args.cuda:
            # self.model = DataParallel(self.model).cuda()
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            if args.resume_pretrain:
                args.start_epoch = checkpoint['epoch']
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.best_pred = checkpoint['best_pred']

            #simplfy/from scratch
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)

            if args.cuda:
                # self.model.module.load_state_dict(model_dict)
                self.model.load_state_dict(model_dict)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        self.lengthtrain = len(self.train_loader)

        self.writer = SummaryWriter(args.save_dir)
    # 训练
    def train(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)

        for i, sample_batched in enumerate(tbar):

            target = sample_batched['label'].type(torch.LongTensor)#target,256 256
            skull = sample_batched['skull'].type(torch.LongTensor)#skull,256 256
            if self.args.cuda:
                target, skull= target.cuda(), skull.cuda()

            # 三维mri图像中抽取的2s张切片
            image_3D = sample_batched['image_stack'].type(torch.FloatTensor)
            image_3D = image_3D.cuda()

            image_2D = sample_batched['image'].type(torch.FloatTensor)
            image_2D = image_2D.cuda()


            out_fuse,out_label,out_skull = self.model(image_3D,image_2D) #N class H W
            outputs=tuple([out_fuse,out_label,out_skull])
            loss = self.criterion(outputs, target,skull )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler(self.optimizer, i, epoch, self.best_pred, self.writer)

            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

            del target, skull, image_3D,image_2D

        print("==== Epoch [" + str(epoch) + " / " + str(self.args.epochs) + "] DONE ====")
        print('Loss: %.3f' % train_loss)
        self.writer.add_scalar('train/eph_tloss', train_loss, epoch)
        self.writer.add_scalar('train/eph_Mloss', train_loss / self.lengthtrain, epoch)
        if self.args.no_val:
            # save checkpoint every epoch
            torch.save({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, os.path.join(self.args.save_dir, 'checkpoint_%03d.pth.tar' % epoch))
            print("save model on epoch %d" % epoch)
    # 验证
    def validation(self, epoch):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        volume_dice_score_list = []
        volume_iou_score_list = []
        batch_size = self.args.test_batch_size #default=1

        with torch.no_grad():
            for ind, sample_batched in enumerate(tbar):
                volume = sample_batched['image_3D'].type(torch.FloatTensor)
                skull = sample_batched['skull_3D'].type(torch.LongTensor)
                labelmap = sample_batched['label_3D'].type(torch.LongTensor)
                volume = torch.squeeze(volume)
                skull= torch.squeeze(skull)
                labelmap = torch.squeeze(labelmap)#256 256 256
                sample_name = sample_batched['name']
                if self.args.cuda:
                    volume, skull,labelmap = volume.cuda(), skull.cuda(),labelmap.cuda()

                z_ax, x_ax, y_ax = np.shape(volume)

                volume_prediction = []
                # i(0-255)
                for i in range(0, len(volume), batch_size):
                    image_2D = volume[i, :, :][np.newaxis, :, :]
                    # 1*256*256 ->1*1*256*256
                    image_2D = image_2D.unsqueeze(0)
                    if i <= int(self.args.num_slices * 2):
                        image_stack01 = volume[0:int(i), :, :][None]
                        image_stack02 = volume[int(i + 1):int(self.args.num_slices * 2 + 1), :, :][None]

                    elif i == int(self.args.num_slices * 2 + 1):
                        image_stack01 = volume[0:int(i - 1), :, :][None]
                        image_stack02 = volume[int(i + 1):int(self.args.num_slices * 2 + 1), :, :][None]

                    elif i >= 245 and i < 255:
                        image_stack01 = volume[z_ax - int(self.args.num_slices * 2 + 1):int(i), :, :][None]
                        image_stack02 = volume[int(i + 1):, :, :][None]

                    elif i == 255:
                        image_stack01 = volume[z_ax - int(self.args.num_slices * 2 + 1):-1, :, :][None]
                        image_stack02 = volume[i + 1:, :, :][None]

                    else:
                        image_stack01 = volume[i - self.args.num_slices:i, :, :][None]
                        image_stack02 = volume[i + 1:i + self.args.num_slices + 1, :, :][None]

                    # 1 10 256 256
                    image_3D = torch.cat((image_stack01, image_stack02), dim=1)

                    out_fuse,out_label,out_skull = self.model(image_3D,image_2D)
                    pred = out_fuse
                    _, batch_output = torch.max(pred, dim=1)
                    volume_prediction.append(batch_output)

                # volume and label are both CxHxW
                volume_prediction = torch.cat(volume_prediction) #one sample_pred 1 256 256 256

                # dice and iou evaluation
                volume_dice_score, volume_iou_score = score_perclass(volume_prediction, labelmap,self.num_class)

                volume_dice_score = volume_dice_score.cpu().numpy()
                volume_dice_score_list.append(volume_dice_score)
                tbar.set_description('Validate Dice Score: %.3f' % (np.mean(volume_dice_score)))

                volume_iou_score = volume_iou_score.cpu().numpy()
                volume_iou_score_list.append(volume_iou_score)

            del volume_prediction

            dice_score_arr = np.asarray(volume_dice_score_list)
            iou_score_arr = np.asarray(volume_iou_score_list)

            #################################### used for malc_grained ###################
            # if self.args.num_class is 139:
            #     label_list = np.array([4, 11, 23, 30, 31, 32, 35, 36, 37, 38, 39, 40,
            #                            41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55,
            #                            56, 57, 58, 59, 60, 61, 62, 63, 64, 69, 71, 72, 73,
            #                            75, 76, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 112,
            #                            113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
            #                            128, 129, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
            #                            143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
            #                            156, 157, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
            #                            171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
            #                            184, 185, 186, 187, 190, 191, 192, 193, 194, 195, 196, 197, 198,
            #                            199, 200, 201, 202, 203, 204, 205, 206, 207])
            #     total_idx = np.arange(0, len(label_list)) #生成[0,1,2,...,len-1]
            #     ignore = np.array([42, 43, 64, 69]) #创建给定对象的数组
            #
            #     valid_idx = [i + 1 for i in total_idx if label_list[i] not in ignore]
            #     valid_idx = [0] + valid_idx
            #
            #     dice_socre_vali = dice_score_arr[:, valid_idx]
            #     iou_score_vali = iou_score_arr[:, valid_idx]
            # else:
            #     dice_socre_vali = dice_score_arr
            #     iou_score_vali = iou_score_arr

            dice_socre_vali = dice_score_arr
            iou_score_vali = iou_score_arr
            ####################################save best model for dice###################

            avg_dice_score = np.mean(dice_socre_vali)
            std_dice_score = np.std(dice_socre_vali)
            avg_iou_score = np.mean(iou_score_vali)
            std_iou_score = np.std(iou_score_vali)

            self.writer.add_scalar('avg_dice', avg_dice_score, epoch)
            print('Validation:')
            print("Mean of dice score : " + str(avg_dice_score))
            print("Std of dice score : " + str(std_dice_score))
            print("Mean of iou score : " + str(avg_iou_score))
            print("Std of dice score : " + str(std_iou_score))

            if avg_dice_score > self.best_pred:
                np.save(os.path.join(self.args.save_dir, "new_dice_score.npy"), dice_score_arr)  ####存的最优模型对应验证的dice和iou矩阵
                np.save(os.path.join(self.args.save_dir, "new_iou_score.npy"), iou_score_arr)

                self.best_pred = avg_dice_score
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                }, os.path.join(self.args.save_dir, 'checkpoint_%03d.pth.tar' % epoch))
                print("save model on epoch %d" % epoch)


def score_perclass(vol_output, ground_truth, num_classes):
    dice_perclass = torch.zeros(num_classes)
    iou_perclass = torch.zeros(num_classes)
    for i in range(num_classes):
        ##########debug######################
        GT = (ground_truth == i).float()
        Pred = (vol_output == i).float()
        inter = torch.sum(torch.mul(GT, Pred)) ##点乘
        union = torch.sum(GT) + torch.sum(Pred) + 0.0001
        dice_perclass[i] = (2 * torch.div(inter, union)) #除法
        iou_perclass[i] = torch.div(inter, union - inter)
    return dice_perclass, iou_perclass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (dcriterionefault: 8)')
    parser.add_argument('--resume-pretrain', action='store_false', default=RESUME_PRETRAIN,
                        help='for training from scratch, set False, for training resume from model, set True')
    parser.add_argument('--resume', type=str, default=RESUME_PATH,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default=SAVE_DIR, type=str, metavar='SAVE',
                        help='directory to save checkpoint (default: none)')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                        help='directory to load data')
    parser.add_argument('--data-list', type=str, default=DATA_LIST,
                        help='directory to read data list')
    parser.add_argument('--encode3D', action='store_true', default=True,
                        help='directory to read data list')
    parser.add_argument('--se-loss', action='store_false', default=False,
                        help='apply se classification loss')
    parser.add_argument('--use-weights', action='store_false', default=False,
                        help='apply class weights for 2DCE loss')

    # training hyper params
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b-train', '--batch-size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('-b-test', '--test-batch-size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('-num-slices', '--num-slices', default=5, type=int,
                        metavar='N', help='slice thickness for spatial encoding')
    parser.add_argument('-num-class', '--num-class', default=NUM_CLASS, type=int,
                        metavar='N', help='number of classes for segmentation')
    # cuda, seed and loggingevaluator
    parser.add_argument('--cuda', action='store_false', default=True,
                        help='check whether to use cuda')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # optimizer params
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print(args)
    solver = Solver(args, args.num_class)
    print('Starting Epoch:', solver.args.start_epoch)#0
    print('Total Epoches:', solver.args.epochs)#100

    for epoch in range(solver.args.start_epoch, solver.args.epochs): # 0-99
        solver.train(epoch)
        if not solver.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
             solver.validation(epoch)
    solver.writer.close()


if __name__ == '__main__':
    main()
