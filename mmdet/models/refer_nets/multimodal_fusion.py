import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class VL_Concat(nn.Module):
    def __init__(self, hidden_size=512,num_input_channels=256,num_output_channels=256,num_featmaps=5):
        super(VL_Concat, self).__init__()
        self.hidden_size=hidden_size
        self.num_input_channels=num_input_channels
        self.num_output_channels=num_output_channels
        self.num_featmaps=num_featmaps
        self.lang_fcs = nn.ModuleList([nn.Linear(self.hidden_size * 2, self.num_output_channels) for i in range(self.num_featmaps)])
        self.visual_convs = nn.ModuleList([nn.Conv2d(self.num_input_channels, self.num_output_channels, 1, 1, 0) for i in range(self.num_featmaps)])
        self.vl_convs = nn.ModuleList(
            [nn.Conv2d((self.num_output_channels * 2), self.num_output_channels, 1, 1, 0) for i in
             range(self.num_featmaps)])

    def forward(self, visual_feat,context, hidden, embedded):

        # x = self.extract_feat(img)
        # num_imgs = img.size(0)

        visual_feat = list(visual_feat)
        vl_feat_list=[]
        for f_id in range(len(visual_feat)):
            visual_feat_level = visual_feat[f_id]
            batch, channel, height, width = visual_feat_level.size()
            lang_feat_emb=self.lang_fcs[f_id](hidden).view(batch, channel, 1, 1).expand(batch, channel, height, width)
            visual_feat_emb=self.visual_convs[f_id](visual_feat_level)
            vl_fuse = torch.cat((lang_feat_emb, visual_feat_emb), dim=1)
            vl_fuse_emb = self.vl_convs[f_id](vl_fuse)
            vl_feat_list.append(vl_fuse_emb)
        vl_feat_list = tuple(vl_feat_list)

        return vl_feat_list


class VL_Dynamic(nn.Module):
    def __init__(self, hidden_size=512,num_input_channels=256,num_output_channels=256,num_featmaps=5):
        super(VL_Dynamic, self).__init__()
        self.hidden_size=hidden_size
        self.num_input_channels=num_input_channels
        self.num_output_channels=num_output_channels
        self.num_featmaps=num_featmaps
        self.lang_fcs = nn.ModuleList([nn.Linear(self.hidden_size * 2, self.num_output_channels) for i in range(self.num_featmaps)])

    def forward(self, visual_feat,context, hidden, embedded):

        # x = self.extract_feat(img)
        # num_imgs = img.size(0)

        visual_feat = list(visual_feat)
        vl_feat_list=[]
        for f_id in range(len(visual_feat)):
            vl_feat_batch_list = []
            visual_feat_level = visual_feat[f_id]
            batch, channel, height, width = visual_feat_level.size()
            dynamic_lang_filter=F.tanh(self.lang_fcs[f_id](hidden))
            for i in range(batch):
                visual_feat_img=visual_feat_level[i].unsqueeze(0)
                dynamic_lang_filter_per_img=dynamic_lang_filter[i].view(1,-1,1,1)
                response=F.conv2d(visual_feat_img,dynamic_lang_filter_per_img)
                vl_fuse_img=response*visual_feat_img
                vl_feat_batch_list.append(vl_fuse_img)
            vl_feat_list.append(torch.cat(vl_feat_batch_list,dim=0))
        vl_feat_list = tuple(vl_feat_list)

        return vl_feat_list

