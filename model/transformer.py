import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from .drop import DropPath
import copy

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class ActionHeadLinprobe(nn.Module):
    def __init__(self, dim_feat=512, num_classes=60, num_joints=25):
        super(ActionHeadLinprobe, self).__init__()
        self.fc = nn.Linear(dim_feat, num_classes)
        
    def forward(self, feat):
        '''
            Input: (N, M, T, J, C)
        '''
        N, M, T, J, C = feat.shape
        feat = feat.mean(dim=[1,2,3])
        feat = self.fc(feat)
        return feat

class ActionHeadFinetune(nn.Module):
    def __init__(self, dropout_ratio=0., dim_feat=512, num_classes=60, num_joints=25, hidden_dim=2048):
        super(ActionHeadFinetune, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_feat*num_joints, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, feat):
        '''
            Input: (N, M, T, J, C)
        '''
        N, M, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 1, 3, 4, 2)      # (N, M, T, J, C) -> (N, M, J, C, T)
        feat = feat.mean(dim=-1)
        feat = feat.reshape(N, M, -1)           # (N, M, J*C)
        feat = feat.mean(dim=1)
        feat = self.fc1(feat)
        feat = self.bn(feat)
        feat = self.relu(feat)    
        feat = self.fc2(feat)
        return feat

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, seqlen=1):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        x = self.forward_attention(q, k, v)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_attention(self, q, k, v):
        B, _, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1,2).reshape(B, N, C*self.num_heads)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., mlp_out_ratio=1.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # assert 'stage' in st_mode
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_out_dim = int(dim * mlp_out_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       out_features=mlp_out_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, seqlen=1):
        x = x + self.drop_path(self.attn(self.norm1(x), seqlen))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
    
class SkeleEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        dim_in=3,
        dim_feat=256,
        num_frames=120,
        num_joints=25,
        patch_size=1,
        t_patch_size=4,
    ):
        super().__init__()
        assert num_frames % t_patch_size == 0
        num_patches = (
            (num_joints // patch_size) * (num_frames // t_patch_size)
        )
        self.input_size = (
            num_frames // t_patch_size,
            num_joints // patch_size
        )
        print(
            f"num_joints {num_joints} patch_size {patch_size} num_frames {num_frames} t_patch_size {t_patch_size}"
        )

        self.num_joints = num_joints
        self.patch_size = patch_size

        self.num_frames = num_frames
        self.t_patch_size = t_patch_size

        self.num_patches = num_patches

        self.grid_size = num_joints // patch_size
        self.t_grid_size = num_frames // t_patch_size

        kernel_size = [t_patch_size, patch_size]
        self.proj = nn.Conv2d(dim_in, dim_feat, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x):
        _, T, V, _ = x.shape
        x = torch.einsum("ntsc->ncts", x)  # [N, C, T, V]
        
        assert (
            V == self.num_joints
        ), f"Input skeleton size ({V}) doesn't match model ({self.num_joints})."
        assert (
            T == self.num_frames
        ), f"Input skeleton length ({T}) doesn't match model ({self.num_frames})."
        
        x = self.proj(x)
        x = torch.einsum("ncts->ntsc", x)  # [N, T, V, C]
        return x

class Transformer(nn.Module):
    def __init__(self, dim_in=3, num_classes=3, dim_feat=256,
                 depth=5, num_heads=8, mlp_ratio=4,
                 num_frames=120, num_joints=25, patch_size=1, t_patch_size=4,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, protocol='linprobe'):
        super().__init__()

        self.num_classes = num_classes
        self.dim_feat = dim_feat

        self.num_frames = num_frames
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.t_patch_size = t_patch_size
        
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.joints_embed = SkeleEmbed(dim_in, dim_feat, num_frames, num_joints, patch_size, t_patch_size)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(dim_feat)

        if protocol == 'linprobe':
            self.head = ActionHeadLinprobe(dim_feat=dim_feat, num_classes=num_classes)
        elif protocol == 'finetune':
            self.head = ActionHeadFinetune(dropout_ratio=0.3, dim_feat=dim_feat, num_classes=num_classes)
        else:
            raise TypeError('Unrecognized evaluation protocol!')

        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames//t_patch_size, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints//patch_size, dim_feat))
        trunc_normal_(self.temp_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Initialize weights
        self.apply(self._init_weights)
        # --------------------------------------------------------------------------
        self.teacher = Encoder(dim_in=dim_in, dim_feat=dim_feat, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                               num_frames=num_frames, num_joints=num_joints, patch_size=patch_size, t_patch_size=t_patch_size,
                               qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                               drop_path_rate=drop_path_rate, norm_layer=norm_layer, is_teacher=True, protocol=protocol)
        self.student = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x):
        N, C, T, V, M = x.shape
        x2 = x.permute(0, 4, 2, 3, 1).contiguous().view(N * M, T, V, C)
        x2 = self.joints_embed(x2)

        NM, TP, VP, _ = x2.shape
        x = x.permute(0, 4, 2, 3, 1).contiguous().view(N * M, T, V, C)
        # embed skeletons
        # self.teacher= self.set_teacher()
        x = self.teacher.forward_encoder(x, mask_ratio=0.0, motion_aware_tau=0.0)

        x = x.reshape(N, M, TP, VP, -1)
        x = self.head(x)
        return x


class Encoder(nn.Module):
    def __init__(self, dim_in=3, dim_feat=256, decoder_dim_feat=256,
                 depth=5, decoder_depth=5, num_heads=8, mlp_ratio=4,
                 num_frames=120, num_joints=25, patch_size=1, t_patch_size=4,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, norm_skes_loss=False,mask_ratio=0.0, motion_aware_tau=0.0,is_teacher=True,protocol='linprobe'):
        super().__init__()
        # MAE decoder specifics
        self.joints_embed = SkeleEmbed(dim_in, dim_feat, num_frames, num_joints, patch_size, t_patch_size)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(dim_feat)

        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames//t_patch_size, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints//patch_size, dim_feat))
        trunc_normal_(self.temp_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        # Copy the encoder components from the parent without sharing paramete
        
        self.patch_size = patch_size
        self.t_patch_size = t_patch_size
        self.mask_ratio = mask_ratio
        self.motion_aware_tau = motion_aware_tau
        self.is_teacher = is_teacher
        # Initialize weights
        self.apply(self.init_weights)
        self.protocol=protocol
        # Make teacher parameters not trainable
        if self.is_teacher:
            self.mask_ratio = 0.0
            self.motion_aware_tau = 0.0
        if self.protocol =='finetune':
            for param in self.parameters():
                param.requires_grad = True
        elif self.protocol == 'linprobe':
            for param in self.parameters():
                param.requires_grad = True
        else:
            raise TypeError('Unrecognized evaluation protocol!')
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def patchify(self, imgs):
        """
        imgs: (N, T, V, 3)
        x: (N, L, t_patch_size * patch_size * 3)
        """
        NM, T, V, C = imgs.shape
        p = self.patch_size
        u = self.t_patch_size
        assert V % p == 0 and T % u == 0
        VP = V // p
        TP = T // u

        x = imgs.reshape(shape=(NM, TP, u, VP, p, C))
        x = torch.einsum("ntuvpc->ntvupc", x)
        x = x.reshape(shape=(NM, TP * VP, u * p * C))
        return x

    def forward_encoder(self,x, mask_ratio, motion_aware_tau):
        
        # x_orig = self.patchify(x)

        # embed skeletons
        x = self.joints_embed(x)

        NM, TP, VP, _ = x.shape

        # add pos & temp embed
        x = x + self.pos_embed[:, :, :VP, :] + self.temp_embed[:, :TP, :, :]

        x = x.reshape(NM, TP * VP, -1)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)

        return x

    def forward(self, x, mask_ratio=0.0, motion_aware_tau=0.0):
        if self.protocol == 'linprobe':
            with torch.no_grad():
                return self.forward_encoder(x, mask_ratio, motion_aware_tau)
        else:
            return self.forward_encoder(x, mask_ratio, motion_aware_tau)

        # if self.is_teacher:
        #     with torch.no_grad():
        #         return self.forward_encoder(x, mask_ratio, motion_aware_tau)
        # else:
        #     return self.forward_encoder(x, mask_ratio, motion_aware_tau)
