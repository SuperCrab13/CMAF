import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pdb
import numpy as np
from os.path import join
from collections import OrderedDict
from models.model_utils import GRL
from models.TransMIL import TransMIL, MultiheadAttention

from nystrom_attention import NystromAttention


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 4,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class LRBilinearFusion(nn.Module):
    def __init__(self, skip=0, use_bilinear=0, gate1=1, gate2=1, dim1=128, dim2=128,
                 scale_dim1=1, scale_dim2=1, dropout_rate=0.25,
                rank=16, output_dim=4):
        super(LRBilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2
        self.rank = rank
        self.output_dim = output_dim

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1_og+dim2_og if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))


        self.h1_factor = Parameter(torch.Tensor(self.rank, dim1 + 1, output_dim))
        self.h2_factor = Parameter(torch.Tensor(self.rank, dim2 + 1, output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))
        xavier_normal(self.h1_factor)
        xavier_normal(self.h2_factor)
        xavier_normal(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

        #init_max_weights(self)

    def forward(self, vec1, vec2):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            h1 = F.dropout(self.linear_h1(vec1), 0.25)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            h2 = F.dropout(self.linear_h2(vec2), 0.25)
            o2 = self.linear_o2(h2)

        ### Fusion
        DTYPE = torch.cuda.FloatTensor
        _o1 = torch.cat((Variable(torch.ones(1, 1).type(DTYPE), requires_grad=False), o1), dim=1)
        _o2 = torch.cat((Variable(torch.ones(1, 1).type(DTYPE), requires_grad=False), o2), dim=1)
        o1_fusion = torch.matmul(_o1, self.h1_factor)
        o2_fusion = torch.matmul(_o2, self.h2_factor)
        fusion_zy = o1_fusion * o2_fusion
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        return output

class BilinearFusion(nn.Module):
    def __init__(self, skip=0, use_bilinear=0, gate1=1, gate2=1, dim1=128, dim2=128, scale_dim1=1, scale_dim2=1, mmhid=256, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1_og+dim2_og if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), 256), nn.ReLU())
        self.encoder2 = nn.Sequential(nn.Linear(256+skip_dim, mmhid), nn.ReLU())
        #init_max_weights(self)

    def forward(self, vec1, vec2):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)

        ### Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, vec1, vec2), 1)
        out = self.encoder2(out)
        return out


class SNN_Block(nn.Module):
    def __init__(self, dim1, dim2, dropout=0.25):
        super(SNN_Block, self).__init__()
        activation = nn.SELU()
        alpha_dropout = nn.AlphaDropout(dropout)
        self.net = nn.Sequential(nn.Linear(dim1,dim2), activation, alpha_dropout)
        for param in self.net.parameters():
            if len(param.shape)==1:
                nn.init.constant_(param, 0)
            else:
                nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="linear")

    def forward(self, data):
        return self.net(data)


def MLP_Block(dim1, dim2, dropout=0.25):
    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=False))


"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes (experimental usage for multiclass MIL)
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes (experimental usage for multiclass MIL)
"""
class Attn_Net_Gated(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x



class TMIL(nn.Module):
    def __init__(self, n_classes):
        super(TMIL, self).__init__()
        self.pos_layer = PPEG(dim=256)
        self._fc1 = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 256))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=256)
        self.layer2 = TransLayer(dim=256)
        self.norm = nn.LayerNorm(256)
        self._fc2 = nn.Linear(256, self.n_classes)

    def forward(self, **kwargs):
        h = kwargs['x_path']  # [B, n, 1024]

        h = self._fc1(h)  # [B, n, 512]

        # ---->pad
        H = h.shape[0]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:add_length, :]], dim=0)  # [B, N, 512]

        # ---->cls_token
        # B = h.shape[0]
        # cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        cls_token = self.cls_token.cuda()
        h = torch.cat((cls_token, h), dim=0)
        h = h.unsqueeze(0)
        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h = self.norm(h)[:, 0]

        # ---->predict
        logits = self._fc2(h)  # [B, n_classes]

        return logits




"""

"""

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class PorpoiseAMIL(nn.Module):
    def __init__(self, size_arg = "small", n_classes=4, stage_num=10):
        super(PorpoiseAMIL, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=0.25, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        self.classifier = nn.Linear(size[1], n_classes)
        initialize_weights(self)


    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')
        else:
            self.attention_net = self.attention_net.to(device)

        self.classifier = self.classifier.to(device)


    def forward(self, **kwargs):
        h = kwargs['x_path']

        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)

        if 'attention_only' in kwargs.keys():
            if kwargs['attention_only']:
                return A

        A_raw = A
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)
        h  = self.classifier(M)
        stage = self.stage_cls(M)
        return h

    def get_slide_features(self, **kwargs):
        h = kwargs['x_path']

        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)

        if 'attention_only' in kwargs.keys():
            if kwargs['attention_only']:
                return A

        A_raw = A
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)
        return M


### MMF (in the PORPOISE Paper)
class PorpoiseMMF(nn.Module):
    def __init__(self,
        omic_input_dim,
        path_input_dim=1024,
        fusion='bilinear',
        dropout=0.25,
        n_classes=4,
        scale_dim1=8,
        scale_dim2=8,
        gate_path=1,
        gate_omic=1,
        skip=True,
        dropinput=0.10,
        use_mlp=False,
        size_arg = "small",
        ):
        super(PorpoiseMMF, self).__init__()
        self.fusion = fusion
        self.size_dict_path = {"small": [path_input_dim, 512, 256], "big": [1024, 512, 384]}
        # self.size_dict_omic = {'small': [512, 512]}
        self.size_dict_omic = {'small': [2048, 1024, 512, 256]}
        self.n_classes = n_classes

        ### Deep Sets Architecture Construction
        size = self.size_dict_path[size_arg]
        if dropinput:
            fc = [nn.Dropout(dropinput), nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        else:
            fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        ### Constructing Genomic SNN
        if self.fusion is not None:
            if use_mlp:
                Block = MLP_Block
            else:
                Block = SNN_Block

            hidden = self.size_dict_omic['small']
            fc_omic = [Block(dim1=omic_input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            self.fc_omic = nn.Sequential(*fc_omic)

            if self.fusion == 'concat':
                self.mm = nn.Sequential(*[nn.Linear(256*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
            elif self.fusion == 'bilinear':
                self.mm = BilinearFusion(dim1=512, dim2=512, scale_dim1=scale_dim1, gate1=gate_path, scale_dim2=scale_dim2, gate2=gate_omic, skip=skip, mmhid=256)
            elif self.fusion == 'lrb':
                self.mm = LRBilinearFusion(dim1=256, dim2=256, scale_dim1=scale_dim1, gate1=gate_path, scale_dim2=scale_dim2, gate2=gate_omic)
            else:
                self.mm = None

        self.classifier_mm = nn.Linear(size[2], n_classes)


    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')

        if self.fusion is not None:
            self.fc_omic = self.fc_omic.to(device)
            self.mm = self.mm.to(device)

        self.classifier_mm = self.classifier_mm.to(device)

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        A, h_path = self.attention_net(x_path)
        A = torch.transpose(A, 1, 0)
        A_raw = A
        A = F.softmax(A, dim=1)
        h_path = torch.mm(A, h_path)

        x_omic = kwargs['x_omic']
        h_omic = self.fc_omic(x_omic)
        if self.fusion == 'bilinear':
            h_mm = self.mm(h_path, h_omic)
        elif self.fusion == 'concat':
            h_mm = self.mm(torch.cat([h_path, h_omic], axis=1))
        elif self.fusion == 'lrb':
            h_mm  = self.mm(h_path, h_omic) # logits needs to be a [1 x 4] vector
            return h_mm

        h_mm  = self.classifier_mm(h_mm) # logits needs to be a [B x 4] vector
        assert len(h_mm.shape) == 2 and h_mm.shape[1] == self.n_classes


        return h_mm

    def captum(self, h, X):
        A, h = self.attention_net(h)
        A = A.squeeze(dim=2)

        A = F.softmax(A, dim=1)
        M = torch.bmm(A.unsqueeze(dim=1), h).squeeze(dim=1) #M = torch.mm(A, h)
        M = self.rho(M)
        O = self.fc_omic(X)

        if self.fusion == 'bilinear':
            MM = self.mm(M, O)
        elif self.fusion == 'concat':
            MM = self.mm(torch.cat([M, O], axis=1))

        logits  = self.classifier(MM)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        risk = -torch.sum(S, dim=1)
        return risk

class CMAF(nn.Module):
    def __init__(self,
        omic_input_dim,
        path_input_dim=1024,
        fusion='bilinear',
        dropout=0.25,
        n_classes=4,
        scale_dim1=8,
        scale_dim2=8,
        gate_path=1,
        gate_omic=1,
        skip=True,
        dropinput=0.10,
        use_mlp=False,
        size_arg = "small",
        ):
        super(CMAF, self).__init__()
        self.fusion = fusion
        self.size_dict_path = {"small": [path_input_dim, 512, 384], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [512, 384]}
        self.n_classes = n_classes

        ### Deep Sets Architecture Construction
        size = self.size_dict_path[size_arg] 
        hidden = self.size_dict_omic['small']
        if self.fusion is not None:
            if use_mlp:
                Block = MLP_Block
            else:
                Block = SNN_Block
        if dropinput:
            self.fc = nn.Sequential(*[nn.Dropout(dropinput), nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout), nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout), nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        attention_net = TransMIL()
        self.attention_net = attention_net

        ### Constructing Genomic SNN


        if self.fusion is not None:
            fc_omic = [Block(dim1=omic_input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            self.fc_omic = nn.Sequential(*fc_omic)

            if self.fusion == 'concat':
                self.mm = nn.Sequential(
                    *[nn.Linear(384 * 2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
            elif self.fusion == 'bilinear':
                self.mm = BilinearFusion(dim1=384, dim2=384, scale_dim1=scale_dim1, gate1=gate_path,
                                         scale_dim2=scale_dim2, gate2=gate_omic, skip=skip, mmhid=size[2])
            elif self.fusion == 'lrb':
                self.mm = LRBilinearFusion(dim1=256, dim2=256, scale_dim1=scale_dim1, gate1=gate_path,
                                           scale_dim2=scale_dim2, gate2=gate_omic)
            else:
                self.mm = None

        self.GRL_layer = GRL()

        self.bn = nn.BatchNorm1d(1024)
        self.discriminator = nn.Sequential(*[nn.Linear(384, 128), nn.ReLU(), nn.Linear(128, 2)])
        self.classifier_mm = nn.Linear(size[2], n_classes)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc = self.fc.to(device)
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')

        if self.fusion is not None:
            self.fc_omic = self.fc_omic.to(device)
            self.mm = self.mm.to(device)

        self.GRL_layer = self.GRL_layer.to(device)
        self.discriminator = self.discriminator.to(device)
        self.bn = self.bn.to(device)
        self.classifier_mm = self.classifier_mm.to(device)


    def forward(self, alpha, **kwargs):
        
        x_omic = kwargs['x_omic']
        h_omic = self.fc_omic(x_omic)

        x_path = kwargs['x_path']
        h_path = self.fc(self.bn(x_path))

        # multi modal discriminator
        if random.randint(0, 1) == 1:
            modality_label = torch.tensor([0,1])
            reverse_feature = self.GRL_layer(torch.concat([h_path.mean(0).unsqueeze(0), h_omic]))
        else:
            modality_label = torch.tensor([1,0])
            reverse_feature = self.GRL_layer(torch.concat([h_omic, h_path.mean(0).unsqueeze(0)]))

        modality_pred = self.discriminator(reverse_feature)

        h_path = self.attention_net(h_path, h_omic)

        if self.fusion == 'bilinear':
            h_mm = self.mm(h_path, h_omic)
        elif self.fusion == 'concat':
            h_mm = self.mm(torch.cat([h_path, h_omic], axis=1))
        elif self.fusion == 'lrb':
            h_mm = self.mm(h_path, h_omic)  # logits needs to be a [1 x 4] vector
            return h_mm

        surv = self.classifier_mm(h_mm)  # logits needs to be a [B x 4] vector
        assert len(surv.shape) == 2 and surv.shape[1] == self.n_classes

        return surv, modality_pred, modality_label #, rec_l



