# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
from functions import ReverseLayerF
class TDNN_block(nn.Module):
    def __init__(self, in_channels=24, out_channels=512,kernel_size=5,dilation=1):
        super(TDNN_block, self).__init__()
        self.tdnn = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation,bias=True)
        self.tdnnd = nn.BatchNorm1d(out_channels)
        self.tdnnb = nn.Dropout(p=0.5)
        self.nonlinearity = nn.ReLU()
    def forward(self, x):
        x = self.tdnn(x)
        x = self.nonlinearity(x)
        x = self.tdnnd(x)
        x = self.tdnnb(x)
        return x
class TFCTL(nn.Module):
    def __init__(self, input_dim = 24, num_classes=2):
        super(TFCTL, self).__init__()
        self.tdnn1 = TDNN_block(in_channels=input_dim, out_channels=512, kernel_size=5, dilation=1)
        self.tdnn2 = TDNN_block(in_channels=512, out_channels=512, kernel_size=3, dilation=2)
        self.tdnn3 = TDNN_block(in_channels=512, out_channels=512, kernel_size=3, dilation=3)
        self.tdnn4 = TDNN_block(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.tdnn5 = TDNN_block(in_channels=512, out_channels=512, kernel_size=1, dilation=3)

        self.tdnn1_1 = TDNN_block(in_channels=5999, out_channels=512, kernel_size=5, dilation=1)
        self.tdnn2_1 = TDNN_block(in_channels=512, out_channels=512, kernel_size=3, dilation=2)
        self.tdnn3_1 = TDNN_block(in_channels=512, out_channels=512, kernel_size=3, dilation=3)
        self.tdnn4_1 = TDNN_block(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.tdnn5_1 = TDNN_block(in_channels=512, out_channels=512, kernel_size=1, dilation=3)

        self.segment6_1 = nn.Sequential(nn.Linear(512 * 2, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(p=0.3),
                                        nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(p=0.3),
                                        nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(p=0.3))
        self.segment6_3 = nn.Sequential(nn.Linear(512 * 2, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(p=0.3),
                                        nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(p=0.3))
        self.segment6_5 = nn.Sequential(nn.Linear(512 * 4, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(p=0.3))

        self.segment6b = nn.BatchNorm1d(512)
        self.segment6d = nn.Dropout(p=0.3)

        self.segment7b = nn.BatchNorm1d(512)
        self.segment7d = nn.Dropout(p=0.3)
        self.nonlinearity = nn.ReLU()
        self.segment6 = nn.Linear(2048, 512)
        self.segment7 = nn.Linear(512*3, 512)
        self.output = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.d_fc1 = nn.Linear(512, 512)
        self.d_bn1 = nn.BatchNorm1d(512)
        self.d_relu1 = nn.ReLU(True)
        self.d_drop1 = nn.Dropout()
        self.d_fc2 = nn.Linear(512, 256)
        self.d_bn2 = nn.BatchNorm1d(256)
        self.d_relu2 = nn.ReLU(True)
        self.d_drop2 = nn.Dropout()
        self.d_fc3 = nn.Linear(256, 2)
    def forward(self, inputs, alpha):
        inputs_y = inputs.transpose(1, 2)
        tdnn1_out = self.tdnn1(inputs_y)
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        tdnn5_out = self.tdnn5(tdnn4_out)

        tdnn1_out_1 = self.tdnn1_1(inputs)
        tdnn2_out_1 = self.tdnn2_1(tdnn1_out_1)
        tdnn3_out_1 = self.tdnn3_1(tdnn2_out_1)
        tdnn4_out_1 = self.tdnn4_1(tdnn3_out_1)
        tdnn5_out_1 = self.tdnn5_1(tdnn4_out_1)

        mean_1 = torch.mean(tdnn1_out, 2)
        std_1 = torch.std(tdnn1_out, 2)
        mean_3 = torch.mean(tdnn2_out, 2)
        std_3 = torch.std(tdnn2_out, 2)
        mean_5 = torch.mean(tdnn5_out, 2)
        std_5 = torch.std(tdnn5_out, 2)
        mean_5_1 = torch.mean(tdnn5_out_1, 2)
        std_5_1 = torch.std(tdnn5_out_1, 2)

        stat_1 = torch.cat((mean_1, std_1), 1)
        stat_3 = torch.cat((mean_3, std_3), 1)
        stat_5 = torch.cat((mean_5, std_5, mean_5_1, std_5_1), 1)
        segment6_1_out = self.segment6_1(stat_1)
        segment6_3_out = self.segment6_3(stat_3)
        segment6_5_out = self.segment6_5(stat_5)
        ### Stat Pool
        segment7_out = self.segment7(torch.cat((segment6_1_out, segment6_3_out, segment6_5_out), 1))
        reverse_feature = ReverseLayerF.apply(segment7_out, alpha)
        x = self.d_fc1(reverse_feature)
        x = self.d_bn1(x)
        x = self.d_relu1(x)
        x = self.d_drop1(x)
        x = self.d_fc2(x)
        x = self.d_bn2(x)
        x = self.d_relu2(x)
        x = self.d_drop2(x)
        domain_output = self.d_fc3(x)
        predictions = self.output(segment7_out)
        return predictions,segment7_out,domain_output