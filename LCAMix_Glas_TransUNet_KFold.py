"""
对k=0,1,2,3层随机选择一层进行cutout, 当k=0时, cutout在input space, 否则在hidden space.
cutout时候的grid 来自于对superpixel根据feature的大小进行的缩放及随机选择
label不变
总共3个loss, Segmentaion loss; Inpainting Loss with binary mask; Dice Loss
# cutout时候的对应生成图像尺寸大小的label, 该label map中在对应位置进行了cutout, 由实际cutout时候的binary mask(feature大小)放大得到
"""

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from networks.A1220_TransUnet_vit_seg_modeling_3Heads_1Decoder import VisionTransformer as ViT_seg
from networks.A1220_TransUnet_vit_seg_modeling_3Heads_1Decoder import CONFIGS as Configs_ViT_Seg
import utils
import os
import random
import argparse
import numpy as np


from torch.utils.data import DataLoader
from skimage import segmentation
import copy
from scipy import stats
from datasets.Glas_dataset import RandomGenerator,ValGenerator,ImageToImage2D_kfold
from metrics.stream_metrics import StreamSegMetrics, dice_binary_class, IoU_binary_class, IoU_multi_class, dice_multi_class,iou_on_batch

import torch
import sys
import torch.nn as nn
import datetime
from losses.DiceLoss import BinaryDiceLoss
from losses.WeightedFocalSuperpixelLoss_layer_pytorch_seg_update import CosFaceLoss
import torch.nn.functional as F

from sklearn.model_selection import KFold
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
        fileName + 'A0103_GlaS_TranUNet_ours_NoFC.log', path=path)
make_print_to_file(path='./logs')

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='GlaS',
                        choices=[ 'ISIC2018T1', 'MoNuSeg', 'ISIC2017T1', 'GlaS'], help='Name of dataset')
    parser.add_argument("--train_dataset", type=str, default='./datasets/data/GlaS/Train_Folder',
                        help="path to Dataset")
    parser.add_argument("--test_dataset", type=str, default='./datasets/data/GlaS/Test_Folder',
                        help="path to Dataset")


    # Train Options
    parser.add_argument("--RESUME", type=bool, default=False)
    parser.add_argument("--START_EPOCH", type=int, default=0,
                        help="epoch number (default: 30k)")
    parser.add_argument("--NB_EPOCH", type=int, default=300,
                        help="epoch number (default: 30k)")
    parser.add_argument("--warmup_itrs", type=int, default=5,
                        help="epoch number (default: 1k)")
    parser.add_argument("--LR", type=float, default=0.0001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=True, #修改！！！
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch size (default: 4)')
    parser.add_argument("--val_batch_size", type=int, default=16,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--img_size", type=int, default=224) #修改！！！
    parser.add_argument("--kfold", type=int, default=5)

    parser.add_argument("--loss_type", type=str, default='BCE',
                        choices=['cross_entropy', 'focal_loss', 'BCE'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',help="GPU ID")
    parser.add_argument("--gpu_ids", type=list, default=[0],help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1234,
                        help="random seed (default: 1)")
    parser.add_argument("--Burn_prob", type=float, default=0.5)
    parser.add_argument("--cutmix_prob", type=float, default=0.5)
    parser.add_argument("--N_min", type=int, default=200)
    parser.add_argument("--N_max", type=int, default=400)

    parser.add_argument("--model", type=str, default='TranUNet_ours_pretrain',help='model name')

    parser.add_argument("--cosface_scale", type=float, default=64,
                        help='scale term for cosface loss (default: 64)')
    parser.add_argument("--cosface_m", type=float, default=0.8,  #从0.35改变到0.05, loss从32变化到9（一开始）
                        help='margin term for cosface loss (default: 0.05)')
    parser.add_argument("--focal_gamma", type=float, default=3.5,
                        help='margin term for cosface loss (default: 2)')

    parser.add_argument("--loss_weight_seg", type=float, default=0.4)
    parser.add_argument("--loss_weight_rec", type=float, default=0.4)
    parser.add_argument("--loss_weight_local", type=float, default=0.2)

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

            # dice += criteria_dice._show_dice(outputs, labels.float())
            # iou += iou_on_batch(labels.float(), outputs)

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

    if opts.dataset == "Synapse":
        filelists = os.listdir(opts.train_dataset)
    else:
        filelists = os.listdir(opts.train_dataset+"/img")
    filelists = np.array(filelists)
    kfold = opts.kfold
    kf = KFold(n_splits=kfold, shuffle=True, random_state=opts.random_seed)

    for fold, (train_index, val_index) in enumerate(kf.split(filelists)):
        train_filelists = filelists[train_index]
        val_filelists = filelists[val_index]
        print("Fold: {}, Total Nums: {}, train: {}, val: {}".format(fold, len(filelists), len(train_filelists),
                                                                    len(val_filelists)))

        train_tf = RandomGenerator(output_size=[opts.img_size, opts.img_size])
        val_tf = ValGenerator(output_size=[opts.img_size, opts.img_size])
        train_dataset = ImageToImage2D_kfold(opts.train_dataset,
                                             train_tf,
                                             image_size=opts.img_size,
                                             filelists=train_filelists,
                                             task_name=opts.dataset)
        val_dataset = ImageToImage2D_kfold(opts.train_dataset,
                                           val_tf,
                                           image_size=opts.img_size,
                                           filelists=val_filelists,
                                           task_name=opts.dataset)
        train_loader = DataLoader(train_dataset,
                                  batch_size=opts.batch_size,
                                  shuffle=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=opts.val_batch_size,
                                shuffle=True)

        model_name = './checkpoints/A1225_best_%s_%s_3Heads_1Decoder_PRETRAIN_%sFold.pth' %  (opts.model, opts.dataset, fold)
        config_vit = Configs_ViT_Seg['R50-ViT-B_16']
        config_vit.n_classes = 1
        config_vit.n_skip = 3

        config_vit.patches.grid = (int(224 / 16), int(224 / 16))
        if len(opts.gpu_ids) > 1:
            if opts.RESUME:
                model = ViT_seg(config_vit, img_size=224, num_classes=1).cuda()
                model.load_from(weights=np.load(config_vit.pretrained_path))
                model.load_state_dict(torch.load(model_name))
            else:
                model = ViT_seg(config_vit, img_size=224, num_classes=1).cuda()
                model.load_from(weights=np.load(config_vit.pretrained_path))
            model = torch.nn.DataParallel(model)
        else:
            if opts.RESUME:
                model = ViT_seg(config_vit, img_size=224, num_classes=1).cuda()
                model.load_from(weights=np.load(config_vit.pretrained_path))
                model.load_state_dict(torch.load(model_name))
            else:
                model = ViT_seg(config_vit, img_size=224, num_classes=1).cuda()
                model.load_from(weights=np.load(config_vit.pretrained_path))

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
        results_file_name = results_out_dir + tm + 'A0103_GlaS_TranUNet_ours_NoFC_%sFold_results.txt' %  (fold)

        best_dice, best_iou = 0, 0
        for epoch in range(opts.START_EPOCH, opts.NB_EPOCH):
            if epoch <= 4:
                if opts.LR >= 0.00001:
                    warmup_lr = opts.LR * ((epoch + 1) / 5)
                    lr = warmup_lr
                else:
                    lr = opts.LR
            elif 4 < epoch <= 89:
                lr = opts.LR
            elif 89 < epoch <= 119:  # 50
                lr = opts.LR / 2
            elif 119 < epoch <= 149:  # 40
                lr = opts.LR / 4
            elif 149 < epoch <= 179:  # 30
                lr = opts.LR / 8
            elif 179 < epoch <= 199:  # 40
                lr = opts.LR / 10
            elif 199 < epoch <= 219:  # 40
                lr = opts.LR / 20
            elif 219 < epoch <= 249:  # 40
                lr = opts.LR / 50
            elif 249 < epoch <= 269:  # 40
                lr = opts.LR / 80
            elif 269 < epoch <= 299:  # 40
                lr = opts.LR / 100
            elif 299 < epoch <= 319:  # 40
                lr = opts.LR / 500
            elif 319 < epoch <= 350:  # 40
                lr = opts.LR / 1000
            print("current epoch:", epoch, "current lr:", lr)

            optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=opts.weight_decay)

            list_loss, list_loss_seg, list_loss_dice, list_loss_mse_a, list_loss_mse_b, list_loss_local = [], [], [], [], [], []
            for i, (samples, _) in tqdm(enumerate(train_loader)):
                images = samples['image']
                labels = samples['label']

                r = np.random.rand(1)  # 从0-1正太分布数据中返回一个数
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
                    loss_MSE_a = criterion_MSE(F.normalize(torch.mul(images[rand_index], (1 - binary_mask).cuda())),
                                               F.normalize(
                                                   torch.mul(out_rec, (1 - binary_mask).cuda())))
                    """local contrasts across image"""
                    SuperP_label_batch = SuperP_label_batch.to(device, dtype=torch.long)
                    loss_local_cosface = criterion_local(F.normalize(out_local), SuperP_label_batch, ratio_superp_batch)
                    if np.any(np.isnan(loss_local_cosface.cpu().detach().numpy())):
                        print("loss_local_contrast NAN:", loss_local_cosface)
                        # print(F.normalize(Local_feas, dim=1))
                        sys.exit()
                    loss = opts.loss_weight_seg * (loss_seg + loss_dice) + opts.loss_weight_rec * (
                                loss_MSE_a + loss_MSE_b) + opts.loss_weight_local * loss_local_cosface
                    list_loss.append(loss)
                    list_loss_seg.append(opts.loss_weight_seg * loss_seg)
                    list_loss_dice.append(opts.loss_weight_seg * loss_dice)
                    list_loss_mse_a.append(opts.loss_weight_rec * loss_MSE_a)
                    list_loss_mse_b.append(opts.loss_weight_rec * loss_MSE_b)
                    list_loss_local.append(opts.loss_weight_local * loss_local_cosface)

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

            iou, dice = validate(model=model, loader=val_loader, device=device)
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
                    "Fold %d, Epoch %d, Loss=%f, Loss_seg=%f, Loss_dice=%f,Loss_MSE_A=%f, Loss_MSE_B=%f,Loss_LOCAL=%f, mIoU=%f, Dice=%f" %
                    (
                    fold, epoch, list_loss, list_loss_seg, list_loss_dice, list_loss_mse_a, list_loss_mse_b, list_loss_local, iou,
                    dice))
            with open(results_file_name, 'a') as file:
                file.write(
                    'Fold %d, Epoch %d, Loss=%f, Loss_seg=%f, Loss_dice=%f,Loss_MSE_A=%f, Loss_MSE_B=%f,Loss_LOCAL=%f, mIoU=%f, Dice=%f \n ' % (
                        fold, epoch, list_loss, list_loss_seg, list_loss_dice, list_loss_mse_a, list_loss_mse_b, list_loss_local,
                        iou, dice))

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
