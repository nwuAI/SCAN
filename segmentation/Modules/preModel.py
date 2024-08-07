
import copy
import torch
from .modelV  import Backbone

def preBiformer(configs, num_classes=21843,num_slices=5,
                pretrained=True,  **kwargs):
    model =Backbone(configs,
                    num_classes=num_classes,
                    num_slices=num_slices,

                    depth=configs.blocks_depth,
                    embed_dim=configs.biformer_pyramid_fm,
                    mlp_ratios=[3, 3, 3, 3],
                    final_upsample=configs.final_upsample,
                    # ------------------------------
                    drop_path_rate=configs.drop_path_rate,
                    n_win=configs.n_win,
                    kv_downsample_mode='identity',
                    kv_per_wins=[-1, -1, -1, -1],
                    topks=[1, 4, 16, -2],  ### -2
                    side_dwconv=5,  ###LCE_Kernel=5
                    before_attn_dwconv=3,
                    layer_scale_init_value=-1,
                    qk_dims=configs.biformer_pyramid_fm,
                    head_dim=configs.head_dim,
                    param_routing=False,
                    diff_routing=False,
                    soft_routing=False,
                    pre_norm=True,
                    pe=None,)

    if pretrained:
        ###
        pretrained_path = configs.biformer_s_pretrained_path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            # model is in....
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = model.load_state_dict(pretrained_dict, strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained model of Biformer encoder---")

            model_dict = model.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "stages." in k:
                    if int(k[7:8])<2:
                        current_layer_num = 1 - int(k[7:8])
                        current_k = "stages_up." + str(current_layer_num) + k[8:]
                        full_dict.update({current_k: v})

            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            msg = model.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")

    return model