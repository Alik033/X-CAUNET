## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
from vgg import *

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, kernel, pad, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=kernel, stride=1, padding=pad, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, kernel, pad, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, kernel, pad, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
##########################################################################
class CCCA(nn.Module):
    def __init__(self, dim, num_heads, kernel, pad, ffn_expansion_factor, bias, LayerNorm_type):
        super(CCCA, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = C_Attention(dim, num_heads, kernel, pad, bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.pos = nn.Parameter(torch.randn(5,96,256,256))

    def forward(self, x, y):
        y = y + self.pos
        c_a = self.attn(self.norm1(x), self.norm1(y))
        c_a_y = c_a + y
        c_a = self.ffn(self.norm1(c_a_y))
        c_a = c_a + c_a_y
        
        return c_a

class C_Attention(nn.Module):
    def __init__(self, dim, num_heads, kernel, pad, bias):
        super(C_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qk = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.qk_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=kernel, stride=1, padding=pad, groups=dim*2, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=kernel, stride=1, padding=pad, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x, y):
        b,c,h,w = x.shape

        qk = self.qk_dwconv(self.qk(x))
        q,k = qk.chunk(2, dim=1)  

        v = self.v_dwconv(self.v(y))

        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=1, embed_dim=48, kernel = 3, pad = 1, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=kernel, stride=1, padding=pad, bias=bias)
        self.bn = nn.BatchNorm2d(num_features=embed_dim)
        self.relu = nn.PReLU(embed_dim)

    def forward(self, x):
        x = self.relu(self.bn(self.proj(x)))

        return x

class OverlapPatchEmbed_rgb(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed_rgb, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class Conv2D_pxp(nn.Module):

    def __init__(self, in_ch, out_ch, k,s,p):
        super(Conv2D_pxp, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(num_features=out_ch)
        self.relu = nn.PReLU(out_ch)

    def forward(self, input):
        return self.relu(self.bn(self.conv(input)))

######################################################################################################################
class SKFF(nn.Module):
    def __init__(self, in_channels, height=2,reduction=8,bias=False):
        super(SKFF, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]
        

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)
        
        return feats_V

####################################################################################################################
#--------------------Underwater Restormer--------------------------------------------------------------------------
class U_Restormer(nn.Module):
    def __init__(self, 
        inp_channels=1, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(U_Restormer, self).__init__()

        self.conv_3 = Conv2D_pxp(1, 32, 3,1,1)
        self.conv_5 = Conv2D_pxp(1, 32, 5,1,2)
        self.conv_7 = Conv2D_pxp(1, 32, 7,1,3)

        self.conv_1 = Conv2D_pxp(1, 96, 1, 1, 0)
        self.conv_rgb = Conv2D_pxp(3, 96, 1, 1, 0)

        self.fusion = SKFF(96,2)

        self.CA = CCCA(dim=96, num_heads=heads[3], kernel = 3, pad = 1, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)


        self.T1 = TransformerBlock(dim=96, num_heads=heads[3], kernel = 3, pad = 1, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.alpha = self.pos = nn.Parameter(torch.randn(1))
        self.beta = self.pos = nn.Parameter(torch.randn(1))

        self.output_pre = nn.Conv2d(96, 48, kernel_size=1, stride=1, padding=0, bias=bias)
        self.output = nn.Conv2d(48, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        #------------Channel Split---------------------------------------
        R = torch.unsqueeze(inp_img[:,0,:,:], dim=1)    # B*1*128*128
        G = torch.unsqueeze(inp_img[:,1,:,:], dim=1)    # B*1*128*128
        B = torch.unsqueeze(inp_img[:,2,:,:], dim=1)    # B*1*128*128
        #--------------3x3 Conv------------------------------------------
        conv3_r = self.conv_3(R)
        conv3_g = self.conv_3(G)
        conv3_b = self.conv_3(B)
        #--------------5x5 Conv------------------------------------------
        conv5_r = self.conv_5(R)
        conv5_g = self.conv_5(G)
        conv5_b = self.conv_5(B)
        #--------------7x7 Conv------------------------------------------
        conv7_r = self.conv_7(R)
        conv7_g = self.conv_7(G)
        conv7_b = self.conv_7(B)
        #----------------------------------------------------------------
        r_c = torch.cat([conv3_r, conv5_r, conv7_r], dim = 1)
        g_c = torch.cat([conv3_r, conv5_r, conv7_r], dim = 1)
        b_c = torch.cat([conv3_r, conv5_r, conv7_r], dim = 1)
        #------------------------Cross-color-channel Attention-----------
        rb_ca = self.CA(r_c, b_c)
        gb_ca = self.CA(g_c, b_c)
        rgb = self.fusion([rb_ca, gb_ca])
        #----------------------------------------------------------------
        rgb_conv1 = self.conv_rgb(inp_img)
        rgb2 = self.T1(rgb_conv1)
        #----------------------------------------------------------------
        r_conv1 = self.conv_1(R) + rgb
        g_conv1 = self.conv_1(G) + rgb
        b_conv1 = self.conv_1(B) + rgb
        rb2_ca = self.CA(r_conv1, b_conv1)
        gb2_ca = self.CA(g_conv1, b_conv1)
        rgb3 = self.fusion([rb2_ca, gb2_ca])
        #----------------------------------------------------------------
        final_fea = self.alpha * rgb3 + self.beta * rgb2

        ouput_layer = self.output(self.output_pre(final_fea)) + inp_img

        return ouput_layer