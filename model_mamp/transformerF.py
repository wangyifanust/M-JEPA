import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from .drop import DropPath


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

import copy

# Generic encoder class for student and teacher
# predicor class

# class complete_mode:
#   self.student = encoder()
#   self.teacher = encoder() or copy.deepcopy(self.student)
#   self.predictor = predictor() // decoder it is in the model class
class VICRegLoss(nn.Module):
    def __init__(self, sim_coeff=25.0, var_coeff=25.0, cov_coeff=1.0, eps=1e-4):
        super(VICRegLoss, self).__init__()
        self.sim_coeff = sim_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.eps = eps

    def forward(self, x, y):
        # Invariance term (mean squared error)
        sim_loss = F.mse_loss(x, y)

        # Variance term (ensure feature variance > threshold)
        std_x = torch.sqrt(x.var(dim=0) + self.eps)
        std_y = torch.sqrt(y.var(dim=0) + self.eps)
        var_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))

        # Covariance term (decorrelate features)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        cov_x = (x.T @ x) / (x.shape[0] - 1)
        cov_y = (y.T @ y) / (y.shape[0] - 1)
        cov_loss = (cov_x.pow(2).sum() - torch.diagonal(cov_x).pow(2).sum()) / x.shape[1]
        cov_loss += (cov_y.pow(2).sum() - torch.diagonal(cov_y).pow(2).sum()) / y.shape[1]

        # Total loss
        loss = self.sim_coeff * sim_loss + self.var_coeff * var_loss + self.cov_coeff * cov_loss
        return loss
    
def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1)).to(dtype=torch.int64)

        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)

class Model(nn.Module):
    def __init__(self, dim_in=3, dim_feat=256, decoder_dim_feat=256,
                 depth=5, decoder_depth=5, num_heads=8, mlp_ratio=4,
                 num_frames=120, num_joints=25, patch_size=1, t_patch_size=4,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, norm_skes_loss=False):
        super().__init__()
        self.dim_feat = dim_feat

        self.num_frames = num_frames
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.t_patch_size = t_patch_size

        self.norm_skes_loss = norm_skes_loss

        # MAE Encoder specifics
        self.joints_embed = SkeleEmbed(dim_in, dim_feat, num_frames, num_joints, patch_size, t_patch_size)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints // patch_size, dim_feat))
        trunc_normal_(self.pos_embed, std=.02)

        # MAE Decoder specifics
        self.decoder_embed = nn.Linear(dim_feat, decoder_dim_feat, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim_feat))
        trunc_normal_(self.mask_token, std=.02)

        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_dim_feat)

        self.decoder_temp_embed = nn.Parameter(torch.zeros(1, num_frames // t_patch_size, 1, decoder_dim_feat))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints // patch_size, decoder_dim_feat))
        trunc_normal_(self.decoder_temp_embed, std=.02)
        trunc_normal_(self.decoder_pos_embed, std=.02)

        self.decoder_pred = nn.Linear(decoder_dim_feat, dim_feat, bias=True)

        self.apply(self._init_weights)
        self.teacher = Encoder(dim_in, dim_feat, decoder_dim_feat, depth, decoder_depth, num_heads, 
                               mlp_ratio, num_frames, num_joints, patch_size, t_patch_size, qkv_bias, qk_scale, drop_rate, attn_drop_rate, 
                               drop_path_rate, norm_layer, norm_skes_loss, is_teacher=True)
        self.student = copy.deepcopy(self.teacher)
        for param in self.student.parameters():
            param.requires_grad = True
        self.student.is_teacher = False
        self.teacher.is_teacher = True

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def extract_motion(self, x, motion_stride=1):
        x_motion = torch.zeros_like(x)
        x_motion[:, :-motion_stride, :, :] = x[:, motion_stride:, :, :] - x[:, :-motion_stride, :, :]
        x_motion[:, -motion_stride:, :, :] = 0
        return x_motion

    def patchify(self, imgs):
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

    def predictor(self, x, mask):
        B, N_vis, C_in = x.shape
        x = self.decoder_embed(x)

        N_mask = mask.shape[1]
        mask_tokens = self.mask_token.repeat(B, N_mask, 1)
        x_cat = torch.cat([x, mask_tokens], dim=1)

        for blk in self.decoder_blocks:
            x_cat = blk(x_cat)
        x_cat = self.decoder_norm(x_cat)
        x_mask_out = x_cat[:, N_vis:, :]
        x_mask_out = self.decoder_pred(x_mask_out)
        
        return x_mask_out

    def forward_loss(self, student_latent, teacher_latent, target, mask, ids_restore):
        recon_loss_latent = F.mse_loss(student_latent, teacher_latent, reduction='none')
        target = self.patchify(target)

        if self.norm_skes_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        contrastive_loss_original = F.mse_loss(target, target, reduction='none')
        recon_loss = recon_loss_latent.mean()
        return recon_loss, recon_loss_latent.mean(), contrastive_loss_original.mean()

    def forward(self, x, mask_ratio=0.80, motion_stride=1, motion_aware_tau=0.75):
        N, C, T, V, M = x.shape
        x = x.permute(0, 4, 2, 3, 1).contiguous().view(N * M, T, V, C)
        x_motion = self.extract_motion(x, motion_stride)

        teacher_latent, maskteacher, ids_restoreteacher, ids = self.teacher(x, mask_ratio=0.0, motion_aware_tau=0.0)
        student_latent, mask, ids_restore, ids_keep = self.student(x, mask_ratio=mask_ratio, motion_aware_tau=motion_aware_tau)
        
        teacher_latent = F.layer_norm(teacher_latent, (teacher_latent.size(-1),))
        teacher_latent = apply_masks(teacher_latent, mask)
        student_latent_predicted = self.predictor(student_latent, mask)

        loss, loss1, loss2 = self.forward_loss(student_latent_predicted, teacher_latent, x_motion, mask, ids_restore)

        return loss, loss1, loss2

class Encoder(nn.Module):
    def __init__(self, dim_in=3, dim_feat=256, decoder_dim_feat=256,
                 depth=5, decoder_depth=5, num_heads=8, mlp_ratio=4,
                 num_frames=120, num_joints=25, patch_size=1, t_patch_size=4,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, norm_skes_loss=False, mask_ratio=0.0, motion_aware_tau=0.0, is_teacher=True):
        super().__init__()
        self.joints_embed = SkeleEmbed(dim_in, dim_feat, num_frames, num_joints, patch_size, t_patch_size)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(dim_feat)
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames // t_patch_size, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints // patch_size, dim_feat))
        trunc_normal_(self.temp_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self.patch_size = patch_size
        self.t_patch_size = t_patch_size
        self.mask_ratio = mask_ratio
        self.motion_aware_tau = motion_aware_tau
        self.is_teacher = is_teacher
        
        self.apply(self.init_weights)

        if self.is_teacher:
            for param in self.parameters():
                param.requires_grad = False
            self.mask_ratio = 0.0
            self.motion_aware_tau = 0.0
        else:
            for param in self.parameters():
                param.requires_grad = True
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
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

    def motion_aware_random_masking(self, x, x_orig, mask_ratio, tau):
        NM, L, D = x.shape
        _, TP, VP, _ = x_orig.shape
        len_keep = int(L * (1 - mask_ratio))

        x_orig_motion = torch.zeros_like(x_orig)
        x_orig_motion[:, 1:, :, :] = torch.abs(x_orig[:, 1:, :, :] - x_orig[:, :-1, :, :])
        x_orig_motion[:, 0, :, :] = x_orig_motion[:, 1, :, :]
        x_orig_motion = x_orig_motion.mean(dim=[3])
        x_orig_motion = x_orig_motion.reshape(NM, L)

        x_orig_motion = x_orig_motion / (torch.max(x_orig_motion, dim=-1, keepdim=True).values * tau + 1e-10)
        x_orig_motion_prob = F.softmax(x_orig_motion, dim=-1)

        noise = torch.log(x_orig_motion_prob) - torch.log(-torch.log(torch.rand(NM, L, device=x.device) + 1e-10) + 1e-10)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([NM, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio, motion_aware_tau):
        x_orig = self.patchify(x)
        x = self.joints_embed(x)

        NM, TP, VP, _ = x.shape
        x = x + self.pos_embed[:, :, :VP, :] + self.temp_embed[:, :TP, :, :]

        x = x.reshape(NM, TP * VP, -1)
        if motion_aware_tau > 0:
            x_orig = x_orig.reshape(shape=(NM, TP, VP, -1))
            x, mask, ids_restore, ids_keep = self.motion_aware_random_masking(x, x_orig, mask_ratio, motion_aware_tau)
        else:   
            x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)

        for idx, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)

        return x, mask, ids_restore, ids_keep

    def forward(self, x, mask_ratio=0.0, motion_aware_tau=0.0):
        if self.is_teacher:
            with torch.no_grad():
                return self.forward_encoder(x, mask_ratio, motion_aware_tau)
        else:
            return self.forward_encoder(x, mask_ratio, motion_aware_tau)