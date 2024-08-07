
import torch.nn as nn
import Modules.Blockmodule as bm
import torch
import torch.nn.functional as F
from .biformerblock import Block
import Modules.modules as modules
import Modules.scale as scale
# import Modules.SEmodule as se
from .Transformers_utils import DFF_Transformer
from .cam import CAM
from .FEA import GridAttentionBlock2D

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
    def __init__(self, configs, num_classes = 1000,  num_slices = 5,
                depth = [4, 4, 4, 4], embed_dim = [64, 128, 256, 512],
                mlp_ratios = [4, 4, 4, 4], head_dim = 32, qk_scale = None,
                drop_path_rate = 0., drop_rate = 0.,
                representation_size = None, final_upsample = "expand_first",
                    ########
                n_win = 7,
                kv_downsample_mode = 'ada_avgpool',
                kv_per_wins = [2, 2, -1, -1],  #
                topks = [8, 8, -1, -1],  #
                side_dwconv = 5,
                before_attn_dwconv = 3,
                layer_scale_init_value = -1,  #
                qk_dims = [None, None, None, None],
                param_routing = False,
                diff_routing = False,
                soft_routing = False,
                pre_norm = True,
                pe = None,
                pe_stages = [0],
                use_checkpoint_stages = [],
                auto_pad = False,
                    # -----------------------
                kv_downsample_kernels = [4, 2, 1, 1],
                kv_downsample_ratios = [4, 2, 1, 1],  # -> kv_per_win = [2, 2, 2, 1]

                param_attention = 'qkvo',
                mlp_dwconv = True,
                aux_loss=False):

        super(Backbone, self).__init__()

        nheads = [dim // head_dim for dim in qk_dims]  # [2 4 8 16]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # [0,...,0]

        self.aux_loss =aux_loss


        ###—Encoder
        # 2s-image
        params['num_channels'] = int(2 * num_slices) # batch_size 10 256 256
        self.encode3D1 = bm.EncoderBlock(params, se_block_type='NONE')  # input size 256 256 #10
        params['num_channels'] = 64
        self.encode3D2 = bm.EncoderBlock(params, se_block_type='NONE')  # 128 128 #64
        self.bottleneck3D = bm.DenseBlock(params, se_block_type='NONE')  # output size 64 64 #64

        # 1-slice
        params['num_channels'] = 1  # batch_size 11 256 256
        self.encode2D1 =bm. EncoderBlock(params, se_block_type='NONE')  # input size 256 256 #11
        params['num_channels'] = 64
        self.encode2D2 = bm.EncoderBlock(params, se_block_type='NONE')  # 128 128 #64
        self.bottleneck2D =bm.DenseBlock(params, se_block_type='NONE')  # output size 64 64 #64


        self.downsample_layers = nn.ModuleList()
        ### <patch merging> “N C H W”
        for i in range(2):
            downsample_layer = nn.Sequential(
                nn.Conv2d(embed_dim[i], embed_dim[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(embed_dim[i + 1])
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        cur = 0
        for i in range(3):
            stage = nn.Sequential(
                *[Block(dim=embed_dim[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,  ###-1
                        topk=topks[i],  # [1 4 16 -2]
                        num_heads=nheads[i],  # [2 4 8 16]
                        n_win=n_win,  ### 7
                        qk_dim=qk_dims[i],  # [64 128 256 512]
                        qk_scale=qk_scale,  # None
                        kv_per_win=kv_per_wins[i],  # [-1,-1,-1,-1]
                        kv_downsample_ratio=kv_downsample_ratios[i],  #default=[4,2,1,1]
                        kv_downsample_kernel=kv_downsample_kernels[i],
                        kv_downsample_mode=kv_downsample_mode,  #identity
                        param_attention=param_attention,  #default='qkvo'
                        param_routing=param_routing,  # false
                        diff_routing=diff_routing,  # false
                        soft_routing=soft_routing,  # false
                        mlp_ratio=mlp_ratios[i],  # [3 3 3 3]
                        mlp_dwconv=mlp_dwconv,  ### false
                        side_dwconv=side_dwconv,  # 5
                        before_attn_dwconv=before_attn_dwconv,  # 3
                        pre_norm=pre_norm,  # True
                        auto_pad=auto_pad) for j in range(depth[i])],  ### false
            )
            self.stages.append(stage)
            cur += depth[i]
        self.norm = nn.BatchNorm2d(embed_dim[2])

        #DFF#
        dim_s=64
        dim_l=128
        self.dff=DFF_Transformer(dim_l,dim_s)


        self.CAM1=CAM(64)
        self.CAM2 = CAM(64)
        self.CAM3 = CAM(64)

        self.FEA1 = GridAttentionBlock2D(in_channels=64, gating_channels=64,
                                                 inter_channels=64, mode='concatenation',
                                                 sub_sample_factor=(1,1))
        self.FEA2 = GridAttentionBlock2D(in_channels=64, gating_channels=64,
                                        inter_channels=64, mode='concatenation',
                                          sub_sample_factor=(1, 1))


        #CSSE#
        # se_block_type=params['se_block']
        # if se_block_type == se.SELayer.CSE.value:
        #     self.SELayer = se.ChannelSELayer(params['num_filters'])
        #
        # elif se_block_type == se.SELayer.SSE.value:
        #     self.SELayer = se.SpatialSELayer(params['num_filters'])
        #
        # elif se_block_type == se.SELayer.CSSE.value:
        #     self.SELayer = se.ChannelSpatialSELayer(params['num_filters'])
        # else:
        #     self.SELayer = None

        ###Decoder
        self.upsample_layers = nn.ModuleList()
        for i in range(2):
            upsample_layer = nn.Sequential(
                nn.ConvTranspose2d(embed_dim[2-i],
                                   embed_dim[2-i] // 2,
                                   kernel_size=2,
                                   stride=2,
                                   padding=0),
                nn.Conv2d(embed_dim[2-i], embed_dim[2-i] // 2, kernel_size=1, stride=1)
            )
            self.upsample_layers.append(upsample_layer)

        self.stages_up = nn.ModuleList()
        count = 7  # block=[4 4 18 4]// count=11 [4 4 4 4]
        for i_layer in range(2):
                stage_up = nn.Sequential(
                    *[Block(dim=embed_dim[1 - i_layer], drop_path=dp_rates[count - j_],
                            layer_scale_init_value=layer_scale_init_value,  ###-1
                            topk=topks[1 - i_layer],  # [1 4 16 -2]
                            num_heads=nheads[1 - i_layer],  # [2 4 8 16]
                            n_win=n_win,  ### 7
                            qk_dim=qk_dims[1 - i_layer],  # [64 128 256 512]
                            qk_scale=qk_scale,  # None
                            kv_per_win=kv_per_wins[1 - i_layer],  # [-1,-1,-1,-1]
                            kv_downsample_ratio=kv_downsample_ratios[1 - i_layer],  #default=[4,2,1,1]
                            kv_downsample_kernel=kv_downsample_kernels[1 - i_layer],  #
                            kv_downsample_mode=kv_downsample_mode,  #identity
                            param_attention=param_attention,  #default='qkvo'
                            param_routing=param_routing,  # false
                            diff_routing=diff_routing,  # false
                            soft_routing=soft_routing,  # false
                            mlp_ratio=mlp_ratios[1 - i_layer],  # [3 3 3 3]
                            mlp_dwconv=mlp_dwconv,  ### false
                            side_dwconv=side_dwconv,  # 5
                            before_attn_dwconv=before_attn_dwconv,  # 3
                            pre_norm=pre_norm,  # True
                            auto_pad=auto_pad) for j_ in range(depth[1 - i_layer])],  ### false
                )
                self.stages_up.append(stage_up)
                count -= depth[1 - i_layer]
        self.norm_up = nn.BatchNorm2d(embed_dim[0])


        # 单张二维切片 定义其通道数为1
        params['num_channels'] = 128 #16 16 #64
        self.decode3 = bm.DecoderBlock(params, se_block_type='NONE')  # 64 64 #64
        self.decode2 = bm.DecoderBlock(params, se_block_type='NONE')  # 128 128 #64
        self.decode1 = bm.DecoderBlock(params, se_block_type='NONE')  # 256 256 #64

        self.dsv4 = modules.UnetDsv3(128, 16, scale_factor=(256, 256))
        self.dsv3 = modules.UnetDsv3(64, 16, scale_factor=(256, 256))
        self.dsv2 = modules.UnetDsv3(64, 16, scale_factor=(256, 256))
        self.dsv1 = modules.UnetDsv3(64, 16, scale_factor=(256, 256))

        self.scale_att = scale.scale_atten_convblock(64, 4)
        self.conv1 = nn.Sequential(nn.Dropout2d(0.1, False),
                                  nn.Conv2d(64, num_classes, 1))  # label output
        self.conv2 = nn.Sequential(nn.Dropout2d(0.1, False),
                                  nn.Conv2d(64, 2, 1))  # skull-stripping output
        self.conv3= nn.Conv2d(64, num_classes, 1)  # merged label output

        # if aux_loss:
        #     self.aux_out = nn.Conv2d(embed_dim[0], num_classes, kernel_size=1)

    def forward(self, input3D,input2D):
        """
        :param input: X
        :return: probabiliy map
        """


        # for 2s-image
        e13D, out1, ind1 = self.encode3D1.forward(input3D)  # 256 256 64
        e23D, out2, ind2 = self.encode3D2.forward(e13D)  # 128 128 64
        bn3D = self.bottleneck3D.forward(e23D)  #64 64 64

        # for 1-image
        e12D, out2D1, ind2D1 = self.encode2D1.forward(input2D)  # 256 256 64
        e22D, out2D2, ind2D2 = self.encode2D2.forward(e12D)  # 128 128 64
        bn2D = self.bottleneck2D.forward(e22D)  #64 64 64

        fusion1 = torch.max(out1, out2D1)
        fusion2 = torch.max(out2, out2D2)
        fusion3 = torch.max(bn3D, bn2D)

        x=fusion3
        skip=[]
        for i in range(3):
            if i>0:
                x=self.downsample_layers[i-1](x)
            x=self.stages[i](x)
            skip.append(x)
        x = self.norm(x)

        ##DFF##
        skip[1],skip[0]=self.dff(skip[1],skip[0])

        for i in range(2):
            x=(self.upsample_layers[i])[0](x)
            x= torch.cat((x, skip[1-i]), dim=1)
            x=(self.upsample_layers[i])[1](x)
            x=self.stages_up[i](x)
            if i==0:
                bi_up=x
        x = self.norm_up(x)

        ###false
        if self.aux_loss:
            aux_out = self.aux_out(x)
            aux_out = F.interpolate(aux_out, size=input2D.shape[-2:], mode='bilinear', align_corners=True)


        #decoder
        up3= self.decode3.forward(fusion3,x)# 64 64 64
        up3=self.CAM1(up3)

        fusion2=self.FEA1(fusion2,up3)
        up2 = self.decode2.forward(fusion2,up3,ind2D2)# 64 128 128
        up2 = self.CAM2(up2)

        fusion1 = self.FEA2(fusion1, up2)
        up1 = self.decode1.forward(fusion1,up2,ind2D1) # 64 256 256
        up1 = self.CAM3(up1)

        dsv4 = self.dsv4(bi_up)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        dsv_cat = torch.cat([dsv1, dsv2, dsv3,dsv4], dim=1)
        outimage = self.scale_att(dsv_cat)
        out_label = self.conv1(up1)  # n class 256 256
        out_skull = self.conv2(outimage)#2分类

        pre_fuse = up1 * outimage
        out_fuse = self.conv3(pre_fuse)

        if self.aux_loss:
            return [out_label, aux_out]
        else:
            return out_fuse,out_label,out_skull





