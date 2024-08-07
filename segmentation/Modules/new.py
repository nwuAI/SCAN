
import torch.nn as nn
import Blockmodule as bm
import torch
import torch.nn.functional as F
from .tcfmodule.TCFnet_utils import inconv, down_block, up_transformer, down_transformer, SemanticMapFusion, up_block


# 二维切片的参数，2s张和单张的参数
params = {'num_filters': 64,
            'kernel_h': 5,
            'kernel_w': 5,
            'kernel_c': 1,
            'stride_conv': 1,
            'pool': 2,
            'stride_pool': 2,
          # Valid options : NONE, CSE, SSE, CSSE
            'se_block': "CSSE",
            'drop_out': 0.1}

class Backbone(nn.Module):
    def __init__(self, out_channels, num_slices, se_loss=True, base_chan=32, map_size=4,
                 conv_num=[2, 2, 0, 0, 2, 2], trans_num=[0, 1, 2, 2, 2, 1, 0, 0],
                 num_heads=[1, 4, 8, 16, 8, 4, 1, 1],
                 fusion_depth=2, fusion_dim=256, fusion_heads=8,
                 norm=nn.BatchNorm2d, act=nn.GELU, aux_loss=False,attfactor=1):
        super(Backbone, self).__init__()

        self.map_size = map_size
        self.attfactor = attfactor
        chan_num = [2 * base_chan, 4 * base_chan, 8 * base_chan, 4 * base_chan, 2 * base_chan, base_chan]
        # dim_head = [chan_num[i] // num_heads[i] for i in range(8)]
        self.conv = nn.Sequential(nn.Dropout2d(0.1, False),
                               nn.Conv2d(64, out_channels, 1))  # label output

        # 2s张-image encoder.定义其通道数为2s
        params['num_channels'] = int(2 * num_slices)  # batch_size 11 256 256
        self.encode3D1 = bm.EncoderBlock(params, se_block_type='NONE')  # input size 256 256 #11
        params['num_channels'] = 64
        self.encode3D2 = bm.EncoderBlock(params, se_block_type='NONE')  # 128 128 #64
        self.encode3D3 = bm.EncoderBlock(params, se_block_type='NONE')  # 64 64 #64
        # self.encode3D4 = bm.EncoderBlock(params, se_block_type='NONE')  # 32 32 #64
        self.bottleneck3D = bm.DenseBlock(params, se_block_type='NONE')  # output size 16 16 #64

        # 2s张-skull encoder,定义其通道数为2s
        params['num_channels'] = int(2 * num_slices)  # batch_size 11 256 256
        self.encode3Ds1 = bm.EncoderBlock(params, se_block_type='NONE')  # input size 256 256 #11
        params['num_channels'] = 64
        self.encode3Ds2 = bm.EncoderBlock(params, se_block_type='NONE')  # 128 128 #64
        self.encode3Ds3 = bm.EncoderBlock(params, se_block_type='NONE')  # 64 64 #64
        # self.encode3Ds4 = bm.EncoderBlock(params, se_block_type='NONE')  # 32 32 #64
        self.bottlenecks3D = bm.DenseBlock(params, se_block_type='NONE')  # output size 16 16 #64

        # 单张二维切片 定义其通道数为1
        params['num_channels'] = 1  # batch_size 11 256 256
        self.encode2D1 =bm. EncoderBlock(params, se_block_type='NONE')  # input size 256 256 #11
        params['num_channels'] = 64
        self.encode2D2 = bm.EncoderBlock(params, se_block_type='NONE')  # 128 128 #64
        self.encode2D3 = bm.EncoderBlock(params, se_block_type='NONE')  # 64 64 #64
        # self.encode2D4 = bm.EncoderBlock(params, se_block_type='NONE')  # 32 32 #64
        self.bottleneck2D =bm. DenseBlock(params, se_block_type='NONE')  # output size 16 16 #64


        self.encoder5 = down_transformer(chan_num[0], chan_num[1], conv_num[0], map_size=self.map_size,
                                         map_generate=True, factor=4, attfactor=self.attfactor)
        self.encoder6 = down_transformer(chan_num[1], chan_num[2], conv_num[1], map_size=self.map_size,
                                         map_generate=False, factor=2, attfactor=self.attfactor)


        # 单张二维切片 定义其通道数为1
        params['num_channels'] =128 #16 16 #64
        # self.decode4 =bm.DecoderBlock(params, se_block_type='NONE')  # input size  32 32 #64
        self.decode3 = bm.DecoderBlock(params, se_block_type='NONE')  # 64 64 #64
        self.decode2 = bm.DecoderBlock(params, se_block_type='NONE')  # 128 128 #64
        self.decode1 = bm.DecoderBlock(params, se_block_type='NONE')  # 256 256 #64

        self.map_fusion = SemanticMapFusion(chan_num[1:3], fusion_dim, fusion_heads, depth=fusion_depth, norm=norm)

        self.decoder5 = up_transformer(chan_num[2], chan_num[3], conv_num[4],map_size,map_shortcut=True, factor=4,
                                       attfactor=self.attfactor)
        self.decoder4 = up_transformer(chan_num[3], chan_num[4], conv_num[5], map_size,map_shortcut=True, factor=8,
                                       attfactor=1)


        self.aux_loss = aux_loss
        if aux_loss:
            self.aux_out = nn.Conv2d(chan_num[4], out_channels, kernel_size=1)

    def forward(self, input3D, skull3D,input2D):
        """
        :param input: X
       :return: probabiliy map
        """

        # for 2s-image
        e13D, out1, ind1 = self.encode3D1.forward(input3D)  # 256 256 64
        e23D, out2, ind2 = self.encode3D2.forward(e13D)  # 128 128 64
        e33D, out3, ind3 = self.encode3D3.forward(e23D)  # 64 64 64
        # e43D, out4, ind4 = self.encode3D4.forward(e33D)  # 32 32 64
        bn3D = self.bottleneck3D.forward(e33D)  ###16 16 64

        # for 2s-skull
        e1s3D, outs1, inds1 = self.encode3Ds1.forward(skull3D)  # 256 256 64
        e2s3D, outs2, inds2 = self.encode3Ds2.forward(e1s3D)  # 128 128 64
        e3s3D, outs3, inds3 = self.encode3Ds3.forward(e2s3D)  # 64 64 64
        # e4s3D, outs4, inds4 = self.encode3Ds4.forward(e3s3D)  # 32 32 64
        bns3D = self.bottlenecks3D.forward(e3s3D)  ###16 16 64

        # for 1-image
        e12D, out2D1, ind2D1 = self.encode2D1.forward(input2D)  # 256 256 64
        e22D, out2D2, ind2D2 = self.encode2D2.forward(e12D)  # 128 128 64
        e32D, out2D3, ind2D3 = self.encode2D3.forward(e22D)  # 64 64 64
        # e42D, out2D4, ind2D4 = self.encode2D4.forward(e32D)  # 32 32 64
        bn2D = self.bottleneck2D.forward(e32D)  ###16 16 64

        # 特征融合，即将每层编码后的2s-image、2s-skull、1-image特征融合
        # 融合方式，（2s-image）*（2s-skull），之后再与1-image进行最大值融合
        fusion1 = torch.max(out1,out2D1)
        fusion12=fusion1*outs1
        fusion2 = torch.max(out2,out2D2)
        fusion22 = fusion2*outs2
        fusion3 = torch.max(out3,out2D3)#4 64 64 64
        fusion32 = fusion3*outs3
        # fusion4 = torch.max(out4,out2D4)
        # fusion42 = fusion4*outs4
        fusion5 = torch.max(bn3D,bn2D)
        fusion52 = fusion5*bns3D ##4 64 32 32



        x5, map5 = self.encoder5(fusion52)#初始语义映射由0参数得到
        x6, map6 = self.encoder6(x5,map5)
        #
        map_list = [map5, map6]  # B C H W
        map_list = self.map_fusion(map_list)
        #
        out5, semantic_map = self.decoder5(x6, x5, map6, map_list[1])  # B 256 32 32/8 8
        out4, semantic_map = self.decoder4(out5, fusion52, semantic_map, map_list[0])  # 8 128 64 64 ,s:8 128 8 8
        #
        ###false
        if self.aux_loss:
            aux_out = self.aux_out(out4)
            aux_out = F.interpolate(aux_out, size=input2D.shape[-2:], mode='bilinear', align_corners=True)

        # image decoder
        # up4=self.decode4.forward(fusion52,fusion42,ind2D4)# 64 32 32
        up3= self.decode3.forward(out4,fusion32,ind2D3)# 64 64 64
        up2 = self.decode2.forward(up3, fusion22,ind2D2)# 64 128 128
        up1 = self.decode1.forward(up2, fusion12,ind2D1) # 64 256 256

        out_label = self.conv(up1)  # n class 256 256
        # out = self.outc(out1)
        #
        if self.aux_loss:
            return [out_label, aux_out]
        else:
            return out_label
            # return out_label,fusion52,map5,map6,x5,x6,out4





