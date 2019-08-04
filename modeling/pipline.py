import torch
from torch import nn

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Pipline(nn.Module):
    def __init__(self, glabole_model, part_body_model, part_paw_model, num_classes):
        super(Pipline, self).__init__()
        self.in_planes = 2048
        self.glabole = glabole_model
        self.part_body = part_body_model
        self.part_paw = part_paw_model
        self.num_classes = num_classes


        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)                              # no shift
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self,img, img_body, img_part):
        cls_score_g, global_feature = self.glabole(img)
        body_feature = self.part_body(img_body)
        paw_feature = self.part_paw(img_part)

        #feature fuse
        g_b_feature = global_feature + body_feature
        g_p_feature = global_feature + paw_feature

        #bbneck--->id loss
        g_b_feat = self.bottleneck(g_b_feature)
        g_p_feat = self.bottleneck(g_p_feature)
        cls_score_gb = self.classifier(g_b_feat)
        cls_score_gp = self.classifier(g_p_feat)

        return cls_score_g, cls_score_gb, cls_score_gp, global_feature, g_b_feature, g_p_feature

    def load_param(self, trained_path, cpu=False):
        if cpu:
            param_dict = torch.load(trained_path, map_location='cpu')
        else:
            param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

