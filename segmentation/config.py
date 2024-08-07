
import ml_collections
import os
# import wget



def get_biformer_configs():

    cfg = ml_collections.ConfigDict()


    cfg.final_upsample = "expand_first"

    # Bi Transformer Configs
    cfg.biformer_pyramid_fm = [64, 128, 256,512]
    cfg.blocks_depth=[4,4,4]
    cfg.n_win = 8 #default=7 when img_size=224
    cfg.head_dim=32
    cfg.drop_rate=0.0 #used in encoder patchEmbed
    cfg.drop_path_rate = 0.15

    cfg.biformer_s_pretrained_path = './weights/biformer_small_best.pth'



    return cfg

