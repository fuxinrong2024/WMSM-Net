import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_
from .vmamba import VSSM

class MutualFeedback_SC_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        self.c_list = c_list
        self.split_att = split_att
        
        self.spatial_atts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=5, stride=1, padding=2),
                nn.Sigmoid(),
                nn.AdaptiveAvgPool2d(1)
            ) for _ in c_list
        ])
        
        self.channel_atts = nn.ModuleList([
            Channel_Att_With_Guide(c, split_att) for c in c_list
        ])
        
        self.gates = nn.ModuleList([
            AdaptiveScaleFusion(dim=dim) for dim in c_list
        ])

    def forward(self, *feature_maps):
        enhanced_feats = []
        for idx, feat in enumerate(feature_maps):
            B, H, W, C = feat.shape
            feat_cwh = feat.permute(0, 3, 1, 2)
            
            avg_out = torch.mean(feat_cwh, dim=1, keepdim=True)
            max_out, _ = torch.max(feat_cwh, dim=1, keepdim=True)
            spatial_in = torch.cat([avg_out, max_out], dim=1)
            
            spatial_weight = self.spatial_atts[idx][0](spatial_in)
            spatial_guide = self.spatial_atts[idx][2](spatial_weight)
            spatial_feat = (spatial_weight * feat_cwh).permute(0, 2, 3, 1)
            
            channel_feat = self.channel_atts[idx](feat, spatial_guide.squeeze(-1).squeeze(-1))
            
            fuse_feat = spatial_feat + channel_feat
            gated_feat = self.gates[idx](fuse_feat, feat)
            
            enhanced_feats.append(gated_feat)
        
        return enhanced_feats

class WMSM(nn.Module):
    def __init__(self, 
                 input_channels=3, 
                 num_classes=1,
                 depths=[2, 2, 9, 2],
                 depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768],
                 dims_decoder=[768, 384, 192, 96],
                 d_state=16,
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 use_attention_bridge=True,
                 split_att='fc',
                 use_wavelet=True,
                 wavelet_layers=None,
                 wavelet_depth_ratio=0.5,):
        super().__init__()
        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes
        self.use_attention_bridge = use_attention_bridge
        self.vssm_dims = dims
        
        self.use_wavelet = use_wavelet
        if wavelet_layers is None:
            wavelet_layers = [True, True, False, False]

        self.vssm = VSSM(
            patch_size=4,
            in_chans=input_channels,
            num_classes=num_classes,
            depths=depths,
            depths_decoder=depths_decoder,
            dims=dims,
            dims_decoder=dims_decoder,
            d_state=d_state,
            drop_path_rate=drop_path_rate,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False,
            use_wavelet=use_wavelet,
            wavelet_layers=wavelet_layers,
            wavelet_depth_ratio=wavelet_depth_ratio
        )
        
        if self.use_attention_bridge:
            self.attention_bridge = MutualFeedback_SC_Att_Bridge(c_list=dims, split_att=split_att)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        encoder_final_feat, raw_skip_list = self.vssm.forward_features(x)
        
        if self.use_attention_bridge:
            enhanced_skip_list = self.attention_bridge(*raw_skip_list)
        else:
            enhanced_skip_list = raw_skip_list
        
        decoder_feat = self.vssm.forward_features_up(encoder_final_feat, enhanced_skip_list)
        
        final_out = self.vssm.forward_final(decoder_feat)
        return torch.sigmoid(final_out) if self.num_classes == 1 else final_out

    def load_from(self):
        if self.load_ckpt_path is None:
            return
        
        try:
            checkpoint = torch.load(self.load_ckpt_path, map_location='cpu')
            pretrained_dict = checkpoint.get('model', checkpoint)
            model_dict = self.vssm.state_dict()
            
            encoder_keys = [k for k in pretrained_dict.keys() if 'layers.' in k or 'patch_embed.' in k]
            encoder_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k in encoder_keys}
            model_dict.update(encoder_dict)
            
            decoder_dict = {}
            for k, v in pretrained_dict.items():
                if 'layers.' in k:
                    layer_idx = int(k.split('.')[1])
                    mapped_idx = 3 - layer_idx
                    mapped_k = k.replace(f'layers.{layer_idx}', f'layers_up.{mapped_idx}')
                    if mapped_k in model_dict:
                        decoder_dict[mapped_k] = v
            model_dict.update(decoder_dict)
            
            self.vssm.load_state_dict(model_dict)
        
        except Exception as e:
            raise e