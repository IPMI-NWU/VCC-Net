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
