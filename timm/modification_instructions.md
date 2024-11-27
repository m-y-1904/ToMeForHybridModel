# Instructions for Modifying ToMe

This document provides step-by-step instructions to manually modify the timm library.
Ensure that you have cloned the original repository as described in the [README.md](./README.md) file.


## 変更手順
#### 1. Open timm/models/mobilevit.py in pytorch-image-models repository
#### 2. Locate the following sections in the file
The class : class MobileVitBlock
#### 3. Replace the existing code in these sections with the code below:
Please replace the code including global variables.

```python
# modified code of class MobileVitBlock
tomeBlockNum=1 #Number of "Block" to introduce Token Merging. Not number of "Layer" to introduce Token Merging! 
tomeBlockCount=0 #Do not change
@register_notrace_module
class MobileVitBlock(nn.Module):
    """ MobileViT block
        Paper: https://arxiv.org/abs/2110.02178?context=cs.LG
    """
    def __init__(
            self,
            in_chs: int,
            out_chs: Optional[int] = None,
            kernel_size: int = 3,
            stride: int = 1,
            bottle_ratio: float = 1.0,
            group_size: Optional[int] = None,
            dilation: Tuple[int, int] = (1, 1),
            mlp_ratio: float = 2.0,
            transformer_dim: Optional[int] = None,
            transformer_depth: int = 2,
            patch_size: int = 8,
            num_heads: int = 4,
            attn_drop: float = 0.,
            drop: int = 0.,
            no_fusion: bool = False,
            drop_path_rate: float = 0.,
            layers: LayerFn = None,
            transformer_norm_layer: Callable = nn.LayerNorm,
            **kwargs,  # eat unused args
    ):
        super(MobileVitBlock, self).__init__()

        layers = layers or LayerFn()
        groups = num_groups(group_size, in_chs)
        out_chs = out_chs or in_chs
        transformer_dim = transformer_dim or make_divisible(bottle_ratio * in_chs)
        
        self.conv_kxk = layers.conv_norm_act(
            in_chs, in_chs, kernel_size=kernel_size,
            stride=stride, groups=groups, dilation=dilation[0])
        self.conv_1x1 = nn.Conv2d(in_chs, transformer_dim, kernel_size=1, bias=False)

        self.transformer = nn.Sequential(*[
            TransformerBlock(
                transformer_dim,
                mlp_ratio=mlp_ratio,
                num_heads=num_heads,
                qkv_bias=True,
                attn_drop=attn_drop,
                proj_drop=drop,
                drop_path=drop_path_rate,
                act_layer=layers.act,
                norm_layer=transformer_norm_layer,
            )
            for _ in range(transformer_depth)
        ])

        self.norm = transformer_norm_layer(transformer_dim)

        self.conv_proj = layers.conv_norm_act(transformer_dim, out_chs, kernel_size=1, stride=1)

        if no_fusion:
            self.conv_fusion = None
        else:
            #change1
            global tomeBlockNum
            global tomeBlockCount
            if tomeBlockCount<tomeBlockNum:
                self.conv_fusion_tome = layers.conv_norm_act(out_chs, out_chs, kernel_size=kernel_size, stride=1)
            else:    
                self.conv_fusion = layers.conv_norm_act(in_chs + out_chs, out_chs, kernel_size=kernel_size, stride=1)
            tomeBlockCount+=1

        self.patch_size = to_2tuple(patch_size)
        self.patch_area = self.patch_size[0] * self.patch_size[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        # Local representation
        x = self.conv_kxk(x)
        x = self.conv_1x1(x)

        # Unfold (feature map -> patches)
        patch_h, patch_w = self.patch_size
        B, C, H, W = x.shape
        new_h, new_w = math.ceil(H / patch_h) * patch_h, math.ceil(W / patch_w) * patch_w
        num_patch_h, num_patch_w = new_h // patch_h, new_w // patch_w  # n_h, n_w
        num_patches = num_patch_h * num_patch_w  # N
        interpolate = False
        if new_h != H or new_w != W:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # [B, C, H, W] --> [B * C * n_h, n_w, p_h, p_w]
        x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [BP, N, C] where P = p_h * p_w and N = n_h * n_w
        x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)

        originally_len_x0 = len(x[0])
        # Global representations
        x = self.transformer(x)
        x = self.norm(x)
        
        if len(x[0]) != originally_len_x0:
            num_token_root = int(math.sqrt(len(x[0])))
            # Fold (patch -> feature map)
            # [B, P, N, C] --> [B*C*n_h, n_w, p_h, p_w]
            x = x.contiguous().view(B, self.patch_area, num_token_root**2, -1)
        
            x = x.transpose(1, 3).reshape(B * C * num_token_root, num_token_root, patch_h, patch_w)
            # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
            x = x.transpose(1, 2).reshape(B, C, num_token_root * patch_h, num_token_root * patch_w)
            if interpolate:
                x = F.interpolate(x, size=(num_token_root*2, num_token_root*2), mode="bilinear", align_corners=False)

            x = self.conv_proj(x)
            
            x = self.conv_fusion_tome(x)
            return x


        # Fold (patch -> feature map)
        # [B, P, N, C] --> [B*C*n_h, n_w, p_h, p_w]
        x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        
        #s = list(range(d.shape[2]))
        #random.shuffle(s)
        #d = d[:,s,:]

        #x=torch.split(x, 5, dim=2)
        #for d in x:
        #    d=torch.flip(d, [0])
        #x = torch.cat(x , dim=2)

        x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        if interpolate:
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        x = self.conv_proj(x)
        if self.conv_fusion is not None:
            x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        return x

```
## Usage Notes
The usage of ToMe remains the same as the original implementation.
Refer to the original timm documentation for further details on its usage.

## Additional Notes
These changes implement ToMe's functionality with additional features, such as adjustable token merging.
Modifications to timm are also required. Ensure that you follow the instructions in the ToMe/ directory to modify the necessary files.
