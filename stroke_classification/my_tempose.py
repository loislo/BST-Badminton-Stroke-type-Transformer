import torch
from torch import nn, Tensor
from positional_encodings.torch_encodings import PositionalEncoding1D
from torchinfo import summary

from stgcn import ST_GCN_10, ST_GCN_12
from tempose import TCN, TransformerEncoder, TransformerLayer, MLP_Head


class GTemPose_TF(nn.Module):
    '''Equal to TemPose_TF adding GCN-based concept.'''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        graph_cfg: dict = {
            'layout': 'coco',
            'strategy': 'spatial'
        },
        edge_importance_weighting=True,
        gcn_data_bn=False,
        gcn_tem_kernel_size=9,
        gcn_dropout=0.5,
        d_model=100, d_head=128, n_head=6, depth_cir=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_cir_kernel_size=5
    ):
        '''`d_model` should be an even number.'''
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        # GCN-based model for poses
        self.gcn_pose = ST_GCN_10(
            in_channels=in_dim,
            num_class=d_model,
            graph_cfg=graph_cfg,
            edge_importance_weighting=edge_importance_weighting,
            data_bn=gcn_data_bn,
            tem_kernel_size=gcn_tem_kernel_size,
            dropout=gcn_dropout
        )

        # TCNs
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_cir_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_cir_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_cir_kernel_size, drop_p)

        # Circumstantial TransformerLayers (from original Temporal TransformerLayers)
        self.learned_token_cir = nn.Parameter(torch.randn(1, d_model))
        self.embedding_cir = nn.Parameter(torch.empty(1, 3, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_cir = TransformerEncoder(d_model, d_head, n_head, depth_cir, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+n_people+3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # MLP Head
        self.mlp_head = MLP_Head(d_model, n_class, d_model * mlp_d_scale, drop_p)

        self.n_people = n_people
        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_cir.squeeze(0))
        self.embedding_cir.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_cir, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        J_only: Tensor,  # J_only: (b*n, c, t, v, m=1)
        pos: Tensor,  # pos: (b, t, n, 2)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        video_len: Tensor  # video_len: (b)
    ):  
        J_only: Tensor = self.gcn_pose(J_only)
        J_only = J_only.view(-1, self.n_people, self.d_model)

        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()
        # pos_top: (b, 2, t)
        # pos_bottom: (b, 2, t)
        # shuttle: (b, 2, t)

        # TCNs
        pos_top: Tensor = self.tcn_top(pos_top)
        pos_bottom: Tensor = self.tcn_bottom(pos_bottom)
        shuttle: Tensor = self.tcn_shuttle(shuttle)
        # pos_top: (b, d, t)
        # pos_bottom: (b, d, t)
        # shuttle: (b, d, t)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        x_cir = torch.stack((pos_top, pos_bottom, shuttle), dim=1)
        b, _, t, d = x_cir.shape  # _ = 3

        # Concat cls token (circumstantial)
        class_token_cir = self.learned_token_cir.view(1, 1, 1, -1).expand(b, 3, -1, -1)
        x_cir = torch.cat((class_token_cir, x_cir), dim=2)
        t += 1

        # Circumstantial embedding
        x_cir = x_cir + self.embedding_cir
        x_cir: Tensor = self.pre_dropout(x_cir)

        # Circumstantial TransformerLayers
        x_cir = x_cir.view(b*3, t, d)

        range_t = torch.arange(0, t, device=x_cir.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, t)
        mask = mask.repeat_interleave(3, dim=0)
        # mask: (b*3, t)

        x_cir = self.encoder_cir(x_cir, mask)
        x_cir = x_cir[:, 0].view(b, 3, d)

        # Concat joints and circumstances
        x: Tensor = torch.cat((J_only, x_cir), dim=1)

        # Concat cls token (interactional)
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)

        # Interactional embedding
        x = x + self.embedding_inter

        # Interactional TransformerLayers
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()

        x = self.mlp_head(x)
        return x


class SpatialTemPose_TF(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2, n_joints=17,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        '''`d_model` should be an even number.'''
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        # Spatial TransformerLayer
        self.project = nn.Linear(in_dim, d_model)
        self.spatial_layer = TransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)
        self.project_2 = nn.Linear(n_joints * d_model, d_model)

        # TCNs
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Common Temporal TransformerLayers
        self.learned_token_ctem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_ctem = nn.Parameter(torch.empty(1, n_people+3, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_ctem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+n_people+3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # MLP Head
        self.mlp_head = MLP_Head(d_model, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_ctem.squeeze(0))
        self.embedding_ctem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_ctem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        J_only: Tensor,  # JnB: (b, t, n, j, input_dim)
        pos: Tensor,  # pos: (b, t, n, 2)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        video_len: Tensor  # video_len: (b)
    ):
        J_only = J_only.transpose(1, 2).contiguous()
        x = self.project(J_only)
        b, n, t, j, d = x.shape

        x = x.view(b*n*t, j, d)
        x = self.spatial_layer(x)
        x = x.view(-1, j*d)
        x = self.project_2(x)
        x = x.view(b, n, t, d)

        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()
        # pos_top: (b, 2, t)
        # pos_bottom: (b, 2, t)
        # shuttle: (b, 2, t)

        # TCNs
        pos_top: Tensor = self.tcn_top(pos_top)
        pos_bottom: Tensor = self.tcn_bottom(pos_bottom)
        shuttle: Tensor = self.tcn_shuttle(shuttle)
        # pos_top: (b, d, t)
        # pos_bottom: (b, d, t)
        # shuttle: (b, d, t)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)
        # x_additional: (b, 3, t, d)

        # Temporal Fusion (TF)
        x = torch.cat((x, x_additional), dim=1)
        n += 3

        # Concat cls token (temporal)
        class_token_tem = self.learned_token_ctem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1

        # Temporal embedding
        x = x + self.embedding_ctem
        x: Tensor = self.pre_dropout(x)

        # Temporal TransformerLayers
        x = x.view(b*n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, t)
        mask = mask.repeat_interleave(n, dim=0)
        # mask: (b*n, t)
        
        x = self.encoder_ctem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Concat cls token (interactional)
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1

        # Interactional embedding
        x = x + self.embedding_inter

        # Interactional TransformerLayers
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()

        x = self.mlp_head(x)
        return x


if __name__ == '__main__':
    # pose: (b*n, c, t, v, m=1)
    # pose = torch.randn((2, 2, 30, 17, 1), dtype=torch.float)
    # pos = torch.randn((1, 30, 2, 2), dtype=torch.float)
    # shuttle = torch.randn((1, 30, 2), dtype=torch.float)
    # videos_len = torch.tensor([30])
    # input_data = [pose, pos, shuttle, videos_len]
    # model = GTemPose_TF(
    #     in_dim=2,
    #     seq_len=30,
    #     n_class=35,
    #     gcn_tem_kernel_size=7,
    #     depth_inter=3
    # )
    # summary(model, input_data=input_data, depth=4, device='cpu')

    n_features = 17
    in_dim = 2
    pose = torch.randn((1, 30, 2, n_features, in_dim), dtype=torch.float)
    pos = torch.randn((1, 30, 2, 2), dtype=torch.float)
    shuttle = torch.randn((1, 30, 2), dtype=torch.float)
    videos_len = torch.tensor([30])
    input_data = [pose, pos, shuttle, videos_len]
    model = SpatialTemPose_TF(
        in_dim=in_dim,
        seq_len=30,
        n_class=35,
        n_joints=n_features,
        d_model=100
    )
    summary(model, input_data=input_data, depth=4, device='cpu')
