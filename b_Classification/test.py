import math
import os
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, \
    f1_score, roc_curve, auc
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAMPlusPlus

from b_Classification.dataset import build_dataset
from b_Classification.models.models_list import vit_s, resnet18, resnet50, resnet101, swin_s
from b_Classification.models.vig import pvig_s_224_gelu


class ReshapeTransform:
    def __init__(self, model):
        pass

    def __call__(self, x):
        result = x[:, 1:, :].reshape(x.size(0), 14, 14, x.size(2))
        result = result.permute(0, 3, 1, 2)
        return result


class ResizeTransform:
    def __init__(self, im_h: int, im_w: int):
        self.height = self.feature_size(im_h)
        self.width = self.feature_size(im_w)

    @staticmethod
    def feature_size(s):
        s = math.ceil(s / 4)  # PatchEmbed
        s = math.ceil(s / 2)  # PatchMerging1
        s = math.ceil(s / 2)  # PatchMerging2
        s = math.ceil(s / 2)  # PatchMerging3
        return s

    def __call__(self, x):
        result = x.reshape(x.size(0), self.height, self.width, x.size(3))
        result = result.permute(0, 3, 1, 2)
        return result


def get_layer(model, model_name):
    if model_name == 'resnet18':
        return [model.resnet.layer4[1].conv2]
    if model_name == 'resnet50':
        return [model.resnet.layer4[2].conv3]
    if model_name == 'resnet101':
        return [model.resnet.layer4[2].conv3]
    if model_name == 'swinT':
        return [model.net.norm]
    if model_name == 'vit':
        return [model.net.blocks[-1].norm1]
    if model_name == 'vig':
        return [model.backbone[-1][0].fc1[1]]


def infer_cls():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True

    # -------------------------------------------------------
    size = 224
    device = torch.device('cuda')
    dataset_name = 'TB' # SIIM, MIMIC, TB, COVID
    label_num = 2  # 2 3 2 3
    model_name = 'vig' # resnet18 resnet50 resnet101 swinT vit vig

    # model = resnet18(label_num)
    # model = resnet50(label_num)
    # model = resnet101(label_num)
    # model = swin_s(label_num)
    # model = vit_s(label_num)
    model = pvig_s_224_gelu(label_num)
    # -------------------------------------------------------

    base = "output/TB_exp_ViG_e5/"
    model_path = base + "0.7831_20_checkpoint.pth"
    # -------------------------------------------------------

    output_dir = base + "Attention map/"
    model.eval().to(device)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.load_state_dict(torch.load(model_path, map_location='cpu')['model'])
    dataset_test = build_dataset(dataset_name, 'test', size)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1)

    pred_list = []
    label_list = []
    pred_score_list = []
    gt = []
    pd = []
    for img, gaze, label, img_path in dataloader_test:
        img = img.to(device)
        label = label.to(device)
        img_name = img_path[0].split('/')[-1]
        # -------------------------------------------------------
        # pred
        # -------------------------------------------------------
        output = model(img)
        _, pred = torch.max(output, 1)
        pred_score = torch.nn.Softmax(dim=1)(output)
        # -------------------------------------------------------
        # attention map
        # -------------------------------------------------------
        target_layer = get_layer(model, model_name)
        if model_name == 'swinT':
            cam = GradCAMPlusPlus(model=model, target_layers=target_layer,
                                  reshape_transform=ResizeTransform(im_h=size, im_w=size))
        elif model_name == 'vit':
            cam = GradCAMPlusPlus(model=model, target_layers=target_layer,
                                  reshape_transform=ReshapeTransform(model))
        else:
            cam = GradCAMPlusPlus(model=model, target_layers=target_layer)
        targets = [ClassifierOutputTarget(label.item())]
        grayscale_cam = cam(input_tensor=img, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        img = Image.open(img_path[0]).resize((size, size))
        img_float_np = np.float32(img) / 255.
        img_float_np = np.expand_dims(img_float_np, axis=-1).repeat(3, axis=-1)

        cam_image = show_cam_on_image(img_float_np, grayscale_cam, use_rgb=True)
        cam_image = Image.fromarray(cam_image)
        cam_image.save(output_dir + img_name)
        # -------------------------------------------------------
        # append to list
        # -------------------------------------------------------
        pred_score_list.append(pred_score[0][1].cpu().detach().numpy().tolist())
        pred_list.append(pred.cpu().detach().numpy().tolist())
        label_list.append(label.cpu().detach().numpy().tolist())
        gt.extend(label.cpu().detach().numpy())
        pd.extend(output.cpu().detach().numpy())

    pred_list = [b for a in pred_list for b in a]
    label_list = [b for a in label_list for b in a]
    # -------------------------------------------------------
    # print metrics
    # -------------------------------------------------------
    acc = accuracy_score(label_list, pred_list)
    print('Accuracy   : {:.4f}'.format(acc))

    precision = precision_score(label_list, pred_list, average='weighted')
    print('Precision  : {:.4f}'.format(precision))

    recall = recall_score(label_list, pred_list, average='weighted')
    print('Recall     : {:.4f}'.format(recall))

    f1 = f1_score(label_list, pred_list, average='weighted')
    print('F1 score   : {:.4f}'.format(f1))

    cm = confusion_matrix(label_list, pred_list)
    print(cm)

    if dataset_name == 'SIIM' or dataset_name == 'TB':
        auc_score = metrics.roc_auc_score(label_list, pred_score_list)
        print('AUC        : {:.4f}'.format(auc_score))
    else:
        gt = np.array(gt)
        pd = np.array(pd)
        gt = label_binarize(np.array(gt), classes=[0, 1, 2])
        fpr = dict()
        tpr = dict()
        roc_auc = []
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(gt[:, i], pd[:, i])
            roc_auc.append(auc(fpr[i], tpr[i]))
        aucavg = np.mean(roc_auc)
        print("AUC: {}".format(roc_auc))
        print("Avg AUC: {}".format(aucavg))



if __name__ == '__main__':
    infer_cls()
