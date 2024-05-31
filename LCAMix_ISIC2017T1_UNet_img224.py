"""
mix以后每个superpixel属于要么background或者foreground, weighted classification
"""

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
# from networks import TinyViT_UNet_Binary_Channel2
from networks import A1213_UNet_binary_3Heads_1Decoder_WithSelfAttention as SegNet
import utils
import os
import random
import argparse
import numpy as np
import sys


from torch.utils.data import DataLoader
from skimage import segmentation
import copy
from scipy import stats
from datasets.ISIC2017T1_dataset import RandomGenerator,ValGenerator,ImageToImage2D
from utils import ext_transforms as et
from metrics.stream_metrics import StreamSegMetrics, dice_binary_class, IoU_binary_class, IoU_multi_class, dice_multi_class,iou_on_batch

import torch
import torch.nn as nn
import datetime
from losses.DiceLoss import BinaryDiceLoss
import torch.nn.functional as F

from losses.WeightedFocalSuperpixelLoss_layer_pytorch_seg import CosFaceLoss
torch.cuda.set_device(0)

def make_print_to_file(path='./'):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    import sys
    import os
    import sys
    import datetime

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime('DAY' + '%Y_%m_%d_')
    sys.stdout = Logger(
        fileName + 'A1213_ISIC2017T1_UNet_NoFold_img224_SuperpixelCutMix_3Heads_1DecodersSA.log', path=path)
make_print_to_file(path='./logs')

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--dataset", type=str, default='ISIC2017T1',
                        choices=['voc', 'cityscapes', 'ISIC2018T1', 'MoNuSeg', 'ISIC2017T1'], help='Name of dataset')
    # parser.add_argument("--train_dataset", type=str, default='./datasets/ISIC2017/train',
    #                     help="path to Dataset")
    # # parser.add_argument("--val_dataset", type=str, default='./datasets/data/MoNuSeg/Test_Folder',
    # #                     help="path to Dataset")
    parser.add_argument("--test_dataset", type=str, default='./datasets/ISIC2017/test',
                        help="path to Dataset")
    parser.add_argument("--train_dataset", type=str, default='./datasets/ISIC2017/train',
                        help="path to Dataset")
    parser.add_argument("--validation_dataset", type=str, default='./datasets/ISIC2017/val',
                        help="path to Dataset")
    # parser.add_argument("--num_classes", type=int, default=2)

    # Train Options
    parser.add_argument("--RESUME", type=bool, default=False)
    parser.add_argument("--START_EPOCH", type=int, default=0,
                        help="epoch number (default: 30k)")
    parser.add_argument("--NB_EPOCH", type=int, default=200,
                        help="epoch number (default: 30k)")
    parser.add_argument("--warmup_itrs", type=int, default=5,
                        help="epoch number (default: 1k)")
    parser.add_argument("--LR", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=True, #修改！！！
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch size (default: 4)')
    parser.add_argument("--val_batch_size", type=int, default=16,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--img_size", type=int, default=224) #修改！！！

    parser.add_argument("--loss_type", type=str, default='BCE',
                        choices=['cross_entropy', 'focal_loss', 'BCE'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',help="GPU ID")
    parser.add_argument("--gpu_ids", type=list, default=[0],help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1234,
                        help="random seed (default: 1)")
    parser.add_argument("--Burn_prob", type=float, default=0.3)
    parser.add_argument("--cutmix_prob", type=float, default=0.7)
    parser.add_argument("--N_min", type=int, default=40)
    parser.add_argument("--N_max", type=int, default=60)

    parser.add_argument("--model", type=str, default='Unet_3Heads',help='model name')

    parser.add_argument("--cosface_scale", type=float, default=64,
                        help='scale term for cosface loss (default: 64)')
    parser.add_argument("--cosface_m", type=float, default=1.0, 
                        help='margin term for cosface loss (default: 0.05)')
    parser.add_argument("--focal_gamma", type=float, default=2.5,
                        help='margin term for cosface loss (default: 2)')

    parser.add_argument("--loss_weight_seg", type=float, default=0.5)
    parser.add_argument("--loss_weight_rec", type=float, default=0.4)
    parser.add_argument("--loss_weight_local", type=float, default=0.1)

    #
    # parser.add_argument("--loss_weight_seg", type=float, default=1)
    # # parser.add_argument("--loss_weight_dice", type=float, default=0.0)
    # parser.add_argument("--loss_weight_local", type=float, default=0.2)
    # parser.add_argument("--temp_local_contrast", type=float, default=0.07)
    # parser.add_argument("--kappa", type=float, default=8)

    return parser


def rand_bnl(size, p_binom=0.5):
    brl = stats.bernoulli.rvs(p_binom, size=size, random_state=None)  # random_state=None指每次生成随机
    # print(brl)
    (zero_idx,) = np.where(brl == int(1))
    return zero_idx


def SuperpixelCutMix(images, labels, N_superpixels_mim, N_superpixels_max, Bern_prob):
    bsz, C, W, H = images.shape
    binary_mask = []
    labels_mix = []
    ratio_superp_batch = []
    SuperP_map_batch_list = []
    SuperP_label_batch = []
    rand_index = torch.randperm(bsz)
    img_a, img_b = images, images[rand_index]
    lb_a, lb_b = labels, labels[rand_index]

    for sp in range(bsz):
        """superpixel for image A"""
        img_seg_a = img_a[sp].reshape(W, H, -1)  # (W,H,C)
        N_a = random.randint(N_superpixels_mim, N_superpixels_max)
        SuperP_map_a = segmentation.slic(img_seg_a.cpu().numpy(), n_segments=N_a, compactness=8, start_label=1)

        """superpixel for image B"""
        img_seg_b = img_b[sp].reshape(W, H, -1)  # (W,H,C)
        N_b = random.randint(N_superpixels_mim, N_superpixels_max)
        SuperP_map_b = segmentation.slic(img_seg_b.cpu().numpy(), n_segments=N_b, compactness=8, start_label=1)
        SuperP_map_b = SuperP_map_b + 10000 # differnt with A

        """randomly select based on image B for mixing """
        SuperP_map_b_value = np.unique(SuperP_map_b)
        sel_region_idx = rand_bnl(p_binom=Bern_prob, size=SuperP_map_b_value.shape[0])

        binary_mask_sp = np.zeros((W, H))
        for v in range(SuperP_map_b_value.shape[0]):
            if v in sel_region_idx:
                bool_v = (SuperP_map_b == SuperP_map_b_value[v])
                binary_mask_sp[bool_v == True] = 1  # mix处mask是1, 否则是0
            else:
                pass

        SuperP_mix_sp = SuperP_map_a * (1 - binary_mask_sp) + SuperP_map_b * binary_mask_sp  #generated mixed superpixel map
        lable_mix_sp = lb_a[sp].cpu().numpy() * (1 - binary_mask_sp) + lb_b[sp].cpu().numpy() * binary_mask_sp #generated mixed label mask

        """study the relationship of the img superpixel and label mask, caculate the pixel nb ratio between the foreground and total"""
        Co_SuperPixel_label = SuperP_mix_sp * lable_mix_sp
        SuperP_mix_sp_value = np.unique(SuperP_mix_sp)
        ratio_superp_sp, label_superp_sp = [], []
        for m in range(SuperP_mix_sp_value.shape[0]):
            bool_m = (SuperP_mix_sp == SuperP_mix_sp_value[m])
            ## total pixel nb under current superpixel
            tt_pixel_nb_per_superp = SuperP_mix_sp[bool_m==True].sum()
            ## foreground pixel nb under current superpixel
            fg_pixel_nb_per_superp = Co_SuperPixel_label[bool_m == True].sum()
            if tt_pixel_nb_per_superp==0:
                ratio = 1.0
            else:
                ratio = fg_pixel_nb_per_superp / tt_pixel_nb_per_superp
            # print("ratio::::", ratio, fg_pixel_nb_per_superp, tt_pixel_nb_per_superp)
            ratio_superp_sp.append(ratio)
            if ratio > 0.5:
                label_superp_sp.append(1)
            else:
                label_superp_sp.append(0)

        SuperP_label_batch.extend(torch.tensor(label_superp_sp))
        ratio_superp_batch.extend(torch.tensor(ratio_superp_sp))
        SuperP_map_batch_list.append(torch.tensor(SuperP_mix_sp))

        binary_mask_sp = torch.tensor(binary_mask_sp)
        binary_mask_ch_sp = copy.deepcopy(binary_mask_sp)
        binary_mask_ch_sp = binary_mask_ch_sp.expand(C, -1, -1)  # torch.Size([3, 32, 32])
        binary_mask.append(binary_mask_ch_sp)

        lable_mix_sp = torch.tensor(lable_mix_sp)
        labels_mix.append(lable_mix_sp)

    ratio_superp_batch = torch.stack(ratio_superp_batch) #
    SuperP_label_batch = torch.stack(SuperP_label_batch)
    labels_mix = torch.stack(labels_mix)
    labels_mix = labels_mix.float()
    binary_mask = torch.stack(binary_mask)
    binary_mask = binary_mask.float()
    images = img_a * (1- binary_mask) + img_b * binary_mask

    return images, labels_mix, binary_mask, ratio_superp_batch, SuperP_label_batch, SuperP_map_batch_list, rand_index



def validate(model, loader, device):
    model.eval()
    with torch.no_grad():
        iou, dice, count = 0, 0, 0
        for i, (samples,_) in tqdm(enumerate(loader)):

            images = samples['image'].to(device, dtype=torch.float32)
            labels = samples['label'].to(device, dtype=torch.long)

            count += images.shape[0]

            outputs = model(images)

            dice += dice_binary_class(outputs, labels)
            iou += IoU_binary_class(outputs, labels)

        dice = dice / count
        iou = iou / count
        # score = metrics.get_results()

    return iou, dice


results_out_dir = './Results_out/'
if not os.path.exists(results_out_dir):
    os.makedirs(results_out_dir)

def main():
    opts = get_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_tf = RandomGenerator(output_size=[opts.img_size, opts.img_size])
    val_tf = ValGenerator(output_size=[opts.img_size, opts.img_size])
    test_tf = ValGenerator(output_size=[opts.img_size, opts.img_size])

    train_dataset = ImageToImage2D(opts.train_dataset,
                                   train_tf,
                                   image_size=opts.img_size)
    val_dataset = ImageToImage2D(opts.validation_dataset,
                                 val_tf,
                                 image_size=opts.img_size)
    test_dataset = ImageToImage2D(opts.test_dataset,
                                  test_tf,
                                  image_size=opts.img_size)

    test_loader = DataLoader(test_dataset,
                             batch_size=opts.val_batch_size,
                             shuffle=False)
    train_loader = DataLoader(train_dataset,
                              batch_size=opts.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=opts.val_batch_size,
                            shuffle=True)
    print("Train, Validation, Test: ", len(train_dataset), len(val_dataset), len(test_dataset))

    model_name = './checkpoints/A1211_best_%s_%s_SuperpixelCutMix_3Heads_1DecoderSA.pth' %  (opts.model, opts.dataset)
    if len(opts.gpu_ids) > 1:
        if opts.RESUME:
            model = SegNet.UNet_3Heads_1Decoder(img_ch=3)
            model.load_state_dict(torch.load(model_name))
            model = model.cuda()
        else:
            model = SegNet.UNet_3Heads_1Decoder(img_ch=3)
            model = model.cuda()
        model = torch.nn.DataParallel(model)
    else:
        if opts.RESUME:
            model = SegNet.UNet_3Heads_1Decoder(img_ch=3)
            model.load_state_dict(torch.load(model_name))
            model = model.cuda()
        else:
            model = SegNet.UNet_3Heads_1Decoder(img_ch=3)
            model = model.cuda()


    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion_seg = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion_seg = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    elif opts.loss_type == 'BCE':
        criterion_seg = nn.BCELoss(reduction='mean')
    criterion_dice = BinaryDiceLoss()
    criterion_MSE = nn.MSELoss()
    criterion_local = CosFaceLoss(64, 2, s=opts.cosface_scale, m=opts.cosface_m,
                            gamma=opts.focal_gamma)

    tm = datetime.datetime.now().strftime('T' + '%m%d%H%M')
    results_file_name = results_out_dir + tm + 'A1213_ISIC2017T1_UNet_NoFold_img224_SuperpixelCutMix_3Heads_1DecoderSA_results.txt'

    best_dice, best_iou = 0.0, 0.0
    for epoch in range(opts.START_EPOCH, opts.NB_EPOCH):
        if epoch <= 4:
            if opts.LR >= 0.00001:
                warmup_lr = opts.LR * ((epoch + 1) / 5)
                lr = warmup_lr
            else:
                lr = opts.LR
        elif 4 < epoch <= 59:
            lr = opts.LR
        elif 59 < epoch <= 79:  # 50
            lr = opts.LR / 2
        elif 79 < epoch <= 99:  # 40
            lr = opts.LR / 4
        elif 99 < epoch <= 119:  # 30
            lr = opts.LR / 8
        elif 119 < epoch <= 139:  # 40
            lr = opts.LR / 10
        elif 139 < epoch <= 159:  # 40
            lr = opts.LR / 20
        elif 159 < epoch <= 179:  # 40
            lr = opts.LR / 40
        elif 179 < epoch <= 199:  # 30
            lr = opts.LR / 80
        elif 199 < epoch <= 219:  # 30
            lr = opts.LR / 100
        elif 219 < epoch <= 239:  # 20
            lr = opts.LR / 200
        elif 239 < epoch <= 259:  # 20
            lr = opts.LR / 400
        elif 259 < epoch <= 279:  # 10
            lr = opts.LR / 800
        elif 279 < epoch <= 289:  # 10
            lr = opts.LR / 1000
        elif 289 < epoch <= 300:  # 10
            lr = opts.LR / 5000
        print("current lr:", lr)

        optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=opts.weight_decay)

        list_loss, list_loss_seg, list_loss_dice, list_loss_mse_a, list_loss_mse_b, list_loss_local = [], [], [], [], [], []
        for i, (samples, _) in tqdm(enumerate(train_loader)):
            images = samples['image']
            labels = samples['label']

            r = np.random.rand(1)  
            if opts.Burn_prob > 0 and r < opts.cutmix_prob:
                # rand_index = torch.randperm(images.shape[0]).cuda()
                # img_a, img_b = images, images[rand_index]
                # lb_a, lb_b = labels, labels[rand_index]
                optimizer.zero_grad()
                images, labels, binary_mask, ratio_superp_batch, SuperP_label_batch, SuperP_map_batch_list, rand_index = \
                    SuperpixelCutMix(images, labels, opts.N_min, opts.N_max, opts.Burn_prob)
                model.train()
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.float32)
                out_seg, out_rec, out_local = model(images, train=True, superpixel_map=SuperP_map_batch_list)
                loss_seg = criterion_seg(out_seg, labels)
                loss_dice = criterion_dice(out_seg, labels)
                loss_MSE_b = criterion_MSE(F.normalize(torch.mul(images, binary_mask.cuda())), F.normalize(
                    torch.mul(out_rec, binary_mask.cuda())))  # MSE loss with binary mask
                loss_MSE_a = criterion_MSE(F.normalize(torch.mul(images[rand_index], (1-binary_mask).cuda())), F.normalize(
                    torch.mul(out_rec, (1-binary_mask).cuda())))
                """local contrasts across image"""
                SuperP_label_batch = SuperP_label_batch.to(device, dtype=torch.long)
                loss_local_cosface = criterion_local(F.normalize(out_local), SuperP_label_batch, ratio_superp_batch)
                if np.any(np.isnan(loss_local_cosface.cpu().detach().numpy())):
                    print("loss_local_contrast NAN:", loss_local_cosface)
                    # print(F.normalize(Local_feas, dim=1))
                    sys.exit()
                loss = opts.loss_weight_seg*(loss_seg + loss_dice) + opts.loss_weight_rec*(loss_MSE_a + loss_MSE_b) + opts.loss_weight_local*loss_local_cosface
                list_loss.append(loss)
                list_loss_seg.append(opts.loss_weight_seg * loss_seg)
                list_loss_dice.append(opts.loss_weight_seg * loss_dice)
                list_loss_mse_a.append(opts.loss_weight_rec * loss_MSE_a)
                list_loss_mse_b.append(opts.loss_weight_rec * loss_MSE_b)
                list_loss_local.append(opts.loss_weight_local*loss_local_cosface)

                # print("loss: ", loss)
                # print("loss_seg: ", opts.loss_weight_seg*(loss_seg + loss_dice))
                # print("loss_MSE: ", opts.loss_weight_rec*(loss_MSE_a + loss_MSE_b))
                # print("loss_local: ", opts.loss_weight_local*loss_local_cosface)

            else:
                optimizer.zero_grad()
                model.train()
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.float32)
                outputs = model(images)
                loss_seg = criterion_seg(outputs, labels)
                loss_dice = criterion_dice(outputs, labels)
                # loss_dice, loss_MSE = torch.zeros(1).cuda(), torch.zeros(1).cuda()
                loss = loss_seg + loss_dice

                list_loss.append(loss)
                list_loss_seg.append(loss_seg)
                list_loss_dice.append(loss_dice)


            loss.backward()
            optimizer.step()

        iou, dice = validate(model=model, loader=test_loader, device=device)
        if len(list_loss) > 0:
            list_loss = torch.stack(list_loss).mean()
        else:
            list_loss = 0

        if len(list_loss_seg) > 0:
            list_loss_seg = torch.stack(list_loss_seg).mean()
        else:
            list_loss_seg = 0

        if len(list_loss_dice) > 0:
            list_loss_dice = torch.stack(list_loss_dice).mean()
        else:
            list_loss_dice = 0

        if len(list_loss_mse_a) > 0:
            list_loss_mse_a = torch.stack(list_loss_mse_a).mean()
        else:
            list_loss_mse_a = 0

        if len(list_loss_mse_b) > 0:
            list_loss_mse_b = torch.stack(list_loss_mse_b).mean()
        else:
            list_loss_mse_b = 0

        if len(list_loss_local) > 0:
            list_loss_local = torch.stack(list_loss_local).mean()
        else:
            list_loss_local = 0


        if dice > best_dice:  # save best model
            best_dice, best_iou = dice, iou
            if len(opts.gpu_ids) > 1:
                torch.save(model.module.state_dict(), model_name)
            else:
                torch.save(model.state_dict(), model_name)
            print(
                "Epoch %d, Loss=%f, Loss_seg=%f, Loss_dice=%f,Loss_MSE_A=%f, Loss_MSE_B=%f,Loss_LOCAL=%f, mIoU=%f, Dice=%f" %
                (epoch, list_loss, list_loss_seg, list_loss_dice,list_loss_mse_a, list_loss_mse_b, list_loss_local, iou, dice))
        with open(results_file_name, 'a') as file:
            file.write(
                'Epoch %d, Loss=%f, Loss_seg=%f, Loss_dice=%f,Loss_MSE_A=%f, Loss_MSE_B=%f,Loss_LOCAL=%f, mIoU=%f, Dice=%f \n ' % (
                    epoch, list_loss, list_loss_seg, list_loss_dice,list_loss_mse_a, list_loss_mse_b, list_loss_local, iou, dice))

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
