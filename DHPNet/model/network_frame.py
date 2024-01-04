import torch
import torch.nn as nn
import torch.nn.functional as F
from .Prototype import *
from .layers import *
from torchvision.models import resnet50

def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class SA_C(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(SA_C, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride

        # ### att
        # ## positional encoding
        pe = self.conv_p(position(h, w, x.is_cuda))

        q_att = q.reshape(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.reshape(b * self.head, self.head_dim, h, w)
        v_att = v.reshape(b * self.head, self.head_dim, h, w)

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        unfold_k = self.unfold(self.pad_att(k_att)).reshape(b * self.head, self.head_dim,
                                                         self.kernel_att * self.kernel_att, h_out,
                                                         w_out)  # b*head, head_dim, k_att^2, h_out, w_out
        unfold_rpe = self.unfold(self.pad_att(pe)).reshape(1, self.head_dim, self.kernel_att * self.kernel_att, h_out,
                                                        w_out)  # 1, head_dim, k_att^2, h_out, w_out

        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(
            1)  # (b*head, head_dim, 1, h_out, w_out) * (b*head, head_dim, k_att^2, h_out, w_out) -> (b*head, k_att^2, h_out, w_out)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).reshape(b * self.head, self.head_dim, self.kernel_att * self.kernel_att,
                                                        h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).reshape(b, self.out_planes, h_out, w_out)

        ## conv
        f_all = self.fc(torch.cat(
            [q.reshape(b, self.head, self.head_dim, h * w), k.reshape(b, self.head, self.head_dim, h * w),
             v.reshape(b, self.head, self.head_dim, h * w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        out_conv = self.dep_conv(f_conv)

        return self.rate1 * out_att + self.rate2 * out_conv

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.resnet50 = resnet50(pretrained=True)

        # Modify the first conv layer to generate fea maps
        self.resnet50.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=1, padding=3, bias=False)

        self.conv1 = self.resnet50.conv1
        self.bn1 = self.resnet50.bn1
        self.relu = self.resnet50.relu

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resnet50.layer2[0].conv2.stride = (2, 2)

        self.layer1 = self.resnet50.layer1
        self.layer2 = self.resnet50.layer2
        self.layer3 = self.resnet50.layer3

    def forward(self, x):
        tensorConv1 = self.conv1(x)
        x = self.bn1(tensorConv1)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        tensorConv2 = x

        x = self.layer2(x)
        tensorConv3 = x

        x = self.layer3(x)

        return x, tensorConv1, tensorConv2, tensorConv3


class CSFF(torch.nn.Module):
    def __init__(self):
        super(CSFF, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
            )

        def Basic_(in_channel, k_att, head, k_conv):
            return SA_C(in_channel, in_channel, k_att, head, k_conv, stride=1, dilation=1)

        def Attention(in_channel):
            return CoordAtt(in_channel, in_channel)

        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels = nc, out_channels=intOutput, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                torch.nn.ReLU(inplace=False)
            )

        self.moduleConv = Basic(1024, 1024)
        self.moduleConv1 = Basic_(1024, 7, 4, 3)
        self.moduleAtt1 = Attention(1024)
        self.moduleUpsample4 = Upsample(1024, 512)

        self.moduleDeconv3 = Basic(1024, 512)
        self.moduleConv2 = Basic_(512, 7, 4, 3)
        self.moduleAtt2 = Attention(512)
        self.moduleUpsample3 = Upsample(512, 256)

        self.moduleDeconv2 = Basic(512, 256)
        self.moduleConv3 = Basic_(256, 7, 4, 3)
        self.moduleAtt3 = Attention(256)
        self.moduleUpsample2 = Upsample(256, 128)

        self.moduleUpsample1 = Basic(192, 128)


    def forward(self, x, skip1, skip2, skip3):
        tensorConv = self.moduleConv(x)
        tensorConv1 = self.moduleConv1(tensorConv)
        tensorAtt1 = self.moduleAtt1(tensorConv)
        tensorConv = tensorAtt1 * tensorConv1
        tensorUpsample4 = self.moduleUpsample4(tensorConv)
        cat4 = torch.cat((skip3, tensorUpsample4), dim=1)

        tensorDeconv3 = self.moduleDeconv3(cat4)
        tensorConv2 = self.moduleConv2(tensorDeconv3)
        tensorAtt2 = self.moduleAtt2(tensorDeconv3)
        tensorDeconv3 = tensorAtt2 * tensorConv2
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        cat3 = torch.cat((skip2, tensorUpsample3), dim=1)

        tensorDeconv2 = self.moduleDeconv2(cat3)
        tensorConv3 = self.moduleConv3(tensorDeconv2)
        tensorAtt3 = self.moduleAtt3(tensorDeconv2)
        tensorDeconv2 = tensorAtt3 * tensorConv3
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        cat2 = torch.cat((skip1, tensorUpsample2), dim=1)

        cat1 = self.moduleUpsample1(cat2)

        return cat1


class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        def decoder(input_channel, output_channel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )

        self.decoder = decoder(128, 3)

    def forward(self, x):
        output = self.decoder(x)
        return output


class frame_net(torch.nn.Module):
    def __init__(self, proto_size=8, feature_dim=1024, proto_dim=1024, shink_thres=0, dropout=0.0):
        super(frame_net, self).__init__()

        self.encoder = Encoder()
        self.mbff = CSFF()
        self.prototype = Prototype(proto_size, feature_dim, proto_dim, shink_thres, dropout)
        self.decoder = Decoder()

    def forward(self, frame, weights=None, mode="train"):
        frame_target = frame[:, -3:, :, :]
        frame_in = frame[:, :-3, :, :]
        fea, skip1, skip2, skip3 = self.encoder(frame_in)


        if mode == "train":
            proto_fea, separation_loss, ortho_loss = self.prototype(fea, fea, fea, fea, mode="train")
            proto_fea = self.mbff(fea, skip1, skip2, skip3)
            output = self.decoder(proto_fea)

            out = dict(output_frame=output, separation_loss=separation_loss, ortho_loss=ortho_loss,
                       frame_target=frame_target)
            return out

        else:
            proto_fea = self.prototype(fea, fea, fea, fea, mode="test")
            proto_fea = self.mbff(fea, skip1, skip2, skip3)
            output = self.decoder(proto_fea)

            out = dict(output_frame=output, frame_target=frame_target)
            return out

    def count_parameters(self):
        a =  sum(p.numel() for p in self.prototype.parameters() if p.requires_grad)
        b =  sum (p.numel() for p in self.encoder.parameters() if p.requires_grad)
        c =  sum(p.numel() for p in self.mbff.parameters() if p.requires_grad)
        return c