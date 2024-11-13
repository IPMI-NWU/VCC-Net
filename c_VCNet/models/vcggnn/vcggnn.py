import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from c_VCGGNN.models.vcggnn.classifier import VisualGuidedClassifier
from c_VCGGNN.models.vcggnn.generator import AttentionMapGenerator


class VCGGNN(nn.Module):
    def __init__(self, num_classes):
        super(VCGGNN, self).__init__()
        # -------------------------------------------------------
        self.classifier = VisualGuidedClassifier(num_classes=num_classes)
        self.generator = AttentionMapGenerator(num_classes=num_classes)
        self.soft_attention = None
        self.hard_attention = None
        self.aux_cls = None
        # -------------------------------------------------------
        save_model = torch.load('models/pretrained_models/pvig_s_82.1.pth.tar')
        model_dict = self.classifier.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.classifier.load_state_dict(model_dict)



    def forward(self, x):
        _, soft_attention, hard_attention, aux_cls  = self.generator(x)
        self.soft_attention = soft_attention
        self.hard_attention = hard_attention
        self.aux_cls = aux_cls
        cls = self.classifier(torch.stack([x, soft_attention / 10.], dim=0))
        return cls

    def get_soft_attention(self):
        return self.soft_attention

    def get_hard_attention(self):
        return self.hard_attention

    def get_aux_cls(self):
        return self.aux_cls

    def get_distance(self):
        return self.classifier.get_dist()


if __name__ == '__main__':
    model = VCGGNN(2)
    input = torch.rand(8, 1, 224, 224)
    pred = model(input)
    soft_attention = model.get_soft_attention()
    hard_attention = model.get_hard_attention()
    aux_cls = model.get_aux_cls()
    print(pred.shape)
    print(soft_attention.shape, hard_attention.shape, aux_cls.shape)
    print(torch.max(soft_attention), torch.max(hard_attention), torch.max(aux_cls))
    print(torch.min(soft_attention), torch.min(hard_attention), torch.min(aux_cls))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    con_feature_dist, con_attention_dist = model.get_distance()
    print(con_feature_dist.shape, con_attention_dist.shape)

    # -------------------------------------------------------
    test_dir = '../../../Data/SIIM-ACR-Gaze/test'
    csv_path = '../../../Data/SIIM-ACR-Gaze/siim_pneumothorax.csv'
    # dataset_test = DatasetGaze_e2e(test_dir, csv_path, 'test', 224)
    # dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1)
    # model = ViG_ViGUNet(num_classes=2)
    # device = torch.device('cuda')
    # base = "../../output/exp_vig_vigunet_12/"
    # model_path = base + "0.872.pth"
    # model.load_state_dict(torch.load(model_path)['model'])
    # model.eval().to(device)
    # for img, gaze_pred, label, path in dataloader_test:
    #     if path[0].split('/')[-1] == '1.2.276.0.7230010.3.1.4.8323329.2239.1517875171.766942.png':
    #         print(path[0])
    #         img = img.to(device)
    #         output = model(img)
    #         print(output)
