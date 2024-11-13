import os
import random
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
from a_Data import misc
from b_Classification.dataset import build_dataset
from c_VCGGNN.models.vcggnn.vcggnn import VCGGNN


def infer_cls():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    seed = 42 + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # -------------------------------------------------------
    size = 224
    device = torch.device('cuda')
    dataset_name = 'SIIM' # SIIM, MIMIC, TB, COVID
    label_num = 2  # 2 3 2 3

    model = VCGGNN(label_num)

    base = "output/SIIM_exp_2e4_10_3/"
    model_path = base + "xxx_checkpoint.pth"
    # -------------------------------------------------------

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_soft_dir):
        os.makedirs(output_soft_dir)
    if not os.path.exists(output_hard_dir):
        os.makedirs(output_hard_dir)

    model.eval().to(device)

    model.load_state_dict(torch.load(model_path, map_location='cpu')['model'])
    dataset_test = build_dataset(dataset_name, 'test', size)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1)

    pred_list = []
    label_list = []
    pred_score_list = []
    gt = []
    pd = []
    mse = 0.
    for img, gaze, label, img_path in dataloader_test:
        img = img.to(device)
        gaze = gaze.to(device)
        label = label.to(device)
        img_name = img_path[0].split('/')[-1]
        # -------------------------------------------------------
        # pred
        # -------------------------------------------------------
        output = model(img)
        soft_attention = model.get_soft_attention()
        hard_attention = model.get_hard_attention()
        _, pred = torch.max(output, 1)
        pred_score = torch.nn.Softmax(dim=1)(output)
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
    mse = mse / len(dataloader_test)
    print(mse)

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
