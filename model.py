import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialEmbedding(nn.Module):
    """NAPL 风格的空间嵌入"""
    def __init__(self, num_nodes, embed_dim):
        super().__init__()
        self.E_spe = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
        
    def forward(self, x):
        # x: [B, C, N, T]
        E_expanded = self.E_spe.transpose(0, 1).unsqueeze(0).unsqueeze(-1)  # [1, C, N, 1]
        return x + E_expanded

class SignalDecompositionLayer(nn.Module):
    """
    将输入信号分解为稳态分量和动态分量
    
    原理：低通滤波提取平滑趋势（稳态），残差为高频波动（动态）
    输入: [B, T, N, F] 其中 F=3: [流量, hod, dow]
    """
    def __init__(self, in_features=1, hidden_dim=32):
        super().__init__()
        
        # 低通滤波器：用 1D 卷积在时间维上平滑（只对流量特征滤波）
        self.low_pass = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=5, padding=2, bias=False),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=5, padding=2, bias=False)
        )
        
        # 可学习的平滑系数
        self.smooth_gate = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        """
        x: [B, T, N, F] 其中 F[:, :, :, 0] = 流量
        
        Returns:
            steady: 稳态分量 [B, T, N, 1]
            dynamic: 动态分量 [B, T, N, 1]
        """
        B, T, N, F = x.shape
        
        # 提取流量特征 [B, T, N, 1]
        flow = x[..., 0:1]
        
        # 低通滤波提取稳态分量
        # 转换为 [B*N, 1, T] 用于 Conv1d
        flow_flat = flow.permute(0, 2, 3, 1).reshape(B * N, 1, T)
        steady_flat = self.low_pass(flow_flat)  # [B*N, 1, T]
        
        # 恢复形状 [B, T, N, 1]
        steady_flow = steady_flat.reshape(B, N, 1, T).permute(0, 3, 1, 2)
        
        # 动态分量 = 原始流量 - 平滑趋势
        dynamic_flow = flow - steady_flow
        
        # 返回稳态和动态分量（只包含流量特征）
        # 时间特征 hod 和 dow 保持不变
        steady = torch.cat([steady_flow, x[..., 1:]], dim=-1)  # [B, T, N, 3]
        dynamic = torch.cat([dynamic_flow, torch.zeros_like(x[..., 1:])], dim=-1)  # [B, T, N, 3]
        
        return steady, dynamic  


class nconv(nn.Module):
    def forward(self, x, A):
        B, C, N, T = x.shape
        
        if A.dim() == 2:
            # 全局共享图：[N, N]
            x = torch.einsum('bcnt,nm->bcmt', (x, A))
            
        elif A.dim() == 3:
            # 每个样本独立的图：[B, N, N]
            x = torch.einsum('bcnt,bnm->bcmt', (x, A))
            
        else:
            raise ValueError(f"Adjacency matrix must be 2D or 3D, got {A.dim()}D")
        
        return x.contiguous()

class SteadyGCN(nn.Module):
    """
    稳态图卷积：先融合多个静态图，再做多阶扩散
    """
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        self.nconv = nconv()
        self.order = order
        self.support_len = support_len
        
        # 可学习的图融合权重
        self.graph_weights = nn.Parameter(torch.ones(support_len) / support_len)
        
        # 输入通道 = (order + 1) * c_in（因为只有一张融合图，不是 support_len 张）
        c_in_total = (order + 1) * c_in
        self.mlp = nn.Conv2d(c_in_total, c_out, kernel_size=(1, 1), bias=True)
        self.dropout = dropout
        
    def forward(self, x, supports):
        """
        x: [B, C, N, T]
        supports: list of adjacency matrices [A_dist, A_conn, A_dtw]
        
        Returns:
            h: [B, c_out, N, T]
        """
        # ===== 先融合图：加权求和 =====
        weights = torch.softmax(self.graph_weights, dim=0)
        A_fused = sum(w * A for w, A in zip(weights, supports))
        
        # ===== 再做多阶扩散 =====
        out = [x]
        
        # 一阶扩散
        x1 = self.nconv(x, A_fused)
        out.append(x1)
        
        # 高阶扩散
        for k in range(2, self.order + 1):
            x2 = self.nconv(x1, A_fused)
            out.append(x2)
            x1 = x2
        
        # 拼接所有阶
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


# ============================================================
# 2. 时间注意力池化
# ============================================================
class TemporalAttentionPooling(nn.Module):
    """
    时间注意力池化：让模型学习每个时间步的重要性权重
    浅层可能关注最近几步（突发），深层可能关注全局（趋势）
    """
    def __init__(self, channels, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(channels // 4, 8)
        
        self.attn = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        """
        x: [B, C, N, T]
        Returns:
            x_weighted: [B, N, C]  加权池化后的节点特征
            attn_weights: [B, N, T]  注意力权重（可用于可视化分析）
        """
        B, C, N, T = x.shape
        
        if T <= 2:
            x_weighted = x.mean(dim=-1)  # [B, C, N]
            x_weighted = x_weighted.permute(0, 2, 1)  # [B, N, C]
            attn_weights = torch.ones(B, N, T, device=x.device) / T
            return x_weighted, attn_weights

        # 转置到 [B*N, T, C]，方便逐节点计算注意力
        x_flat = x.permute(0, 2, 3, 1).reshape(B * N, T, C)
        
        # 计算注意力权重 [B*N, T, 1]
        attn_weights = self.attn(x_flat)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # 加权求和 [B*N, 1, C] → [B*N, C]
        x_weighted = torch.bmm(attn_weights.transpose(1, 2), x_flat).squeeze(1)
        
        # 恢复形状
        x_weighted = x_weighted.reshape(B, N, C)
        attn_weights = attn_weights.reshape(B, N, T)
        
        return x_weighted, attn_weights


# ============================================================
# 3. 双边调制网络（核心组件）
# ============================================================
class BilateralModulation(nn.Module):
    """
    双边调制：节点活跃度 × 节点间相似度
    
    核心思想：
    - activity: "这个节点当前有多特殊？"（0~1）
    - similarity: "这两个节点的特征模式有多像？"（-1~1）
    - 最终调制 = activity_i × activity_j × similarity_{i,j}
    
    只有两个节点都"活跃"时，它们的边才会被显著增强
    """
    def __init__(self, num_nodes, in_channels, hidden_dim=32, top_k=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.top_k = top_k  # 稀疏化参数，保留最强 K 条边
        
        # ---- 节点活跃度网络 ----
        # 输入：节点特征 [B, N, C]
        # 输出：活跃度分数 [B, N, 1]，范围 [0, 1]
        self.activity_net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # ---- 节点间关系网络 ----
        # 将节点特征投影到关系空间，用于计算相似度
        self.relation_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # ---- 可学习的调制强度 ----
        # alpha: 控制动态调制占基础图的比例
        # 初始化为小值（0.1），让模型先依赖基础图，再逐步学习动态调制
        self.alpha = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x_pooled, A_base):
        B, N, C = x_pooled.shape
        
        # ---- Step 1: 活跃度 ----
        activity = self.activity_net(x_pooled)  # [B, N, 1]
        
        # ---- Step 2: 相似度 ----
        x_rel = self.relation_proj(x_pooled)  # [B, N, D]
        similarity = torch.bmm(x_rel, x_rel.transpose(1, 2))
        similarity = similarity / (x_rel.size(-1) ** 0.5)
        
        # ---- Step 3: 双边调制 ----
        activity_outer = torch.bmm(activity, activity.transpose(1, 2))
        dynamic_mod = activity_outer * similarity  # [B, N, N]
        
        # ---- Step 4: 行中心化 + ReLU + 归一化 ----
        # 核心：制造行内差异
        dynamic_mod = dynamic_mod - dynamic_mod.mean(dim=-1, keepdim=True)
        dynamic_mod = F.relu(dynamic_mod)
        # 防止全零行
        row_sum = dynamic_mod.sum(dim=-1, keepdim=True) + 1e-8
        A_dynamic = dynamic_mod / row_sum
        
        # ---- Step 5: 基础图归一化 ----
        A_base_norm = F.softmax(A_base, dim=-1).unsqueeze(0).expand(B, -1, -1)
        
        # ---- Step 6: 混合 ----
        alpha_gate = torch.sigmoid(self.alpha)
        A_modulated = alpha_gate * A_dynamic + (1 - alpha_gate) * A_base_norm
        
        return A_modulated


class SteadyStreamBlock(nn.Module):
    """
    稳态流块（Static Stream Block）
    
    特点：
    - 使用静态图卷积（多图融合 + 多阶扩散）
    - 适合捕获周期性、稳定的空间依赖
    """
    def __init__(self, num_nodes, layer_idx, supports, residual_channels, dilation_channels,
                 skip_channels, out_dim, dropout=0.3,  
                 kernel_size=2, dilation=1):
        super().__init__()
        
        self.support_len = len(supports)
        self.layer_idx = layer_idx
        self.dilation = dilation
        
        # 扩散阶数：按深度分组
        if layer_idx <= 1:
            self.order = 1
        elif layer_idx <= 5:
            self.order = 2
        else:
            self.order = 3
        # ============================================
        # 时间卷积（膨胀卷积门控）
        # ============================================
        self.filter_conv = nn.Conv2d(
            residual_channels, dilation_channels,
            kernel_size=(1, kernel_size), dilation=dilation
        )
        self.gate_conv = nn.Conv2d(
            residual_channels, dilation_channels,
            kernel_size=(1, kernel_size), dilation=dilation
        )
        
        # ============================================
        # 空间卷积：静态图卷积（多图融合 + 多阶扩散）
        # ============================================
        self.gcn = SteadyGCN(
            dilation_channels, residual_channels, dropout,
            support_len=self.support_len, order=self.order
        )
        
        # ============================================
        # 归一化层
        # ============================================
        self.bn = nn.BatchNorm2d(residual_channels)
        
        # ============================================
        # 输出头
        # ============================================
        self.output_conv = nn.Conv2d(
            residual_channels, out_dim, kernel_size=(1, 1)
        )
        
        # ============================================
        # Skip connection
        # ============================================
        self.skip_conv = nn.Conv2d(
            residual_channels, skip_channels, kernel_size=(1, 1)
        )
        
    def forward(self, x, static_supports):
        """
        Args:
            x: [B, C, N, T] 输入特征
            static_supports: list of adjacency matrices
        Returns:
            layer_output: [B, out_dim, N, T] 当前层的输出
            residual: [B, C, N, T] 残差（用于下一层）
        """
        residual_input = x
        
        # ============================================
        # 1. 时间卷积（门控膨胀卷积）
        # ============================================
        filter_x = torch.tanh(self.filter_conv(x))
        gate_x = torch.sigmoid(self.gate_conv(x))
        x_tcn = filter_x * gate_x
        s = x_tcn
        s = self.skip_conv(s)
        
        # ============================================
        # 2. 空间卷积（静态图卷积）
        # ============================================
        x_gcn = self.gcn(x_tcn, static_supports)
        
        # ============================================
        # 3. 残差连接 + 归一化
        # ============================================
        T_gcn = x_gcn.size(3)
        x_out = x_gcn + residual_input[:, :, :, -T_gcn:]
        x_out = self.bn(x_out)
        
        return x_out, s


class DynamicStreamBlock(nn.Module):
    """
    动态流块（Dynamic Stream Block）
    
    特点：
    - 使用输入感知动态图卷积（样本级图）
    - 适合捕获时变、突发性的空间依赖
    """
    def __init__(self, num_nodes, layer_idx, residual_channels, dilation_channels,
                 skip_channels, out_dim, dropout=0.3, 
                 kernel_size=2, dilation=1, top_k=None, mod_hidden_dim=32):
        super().__init__()

         # 存储层信息（用于调试和分析）
        self.layer_idx = layer_idx
        self.dilation = dilation
        # 扩散阶数：按深度分组
        if layer_idx <= 1:
            self.order = 1
        elif layer_idx <= 5:
            self.order = 2
        else:
            self.order = 3
        c_in_total = (self.order + 1) * dilation_channels
        self.mlp = nn.Conv2d(c_in_total, residual_channels, kernel_size=(1, 1), bias=True)
        self.dropout = dropout
        # ============================================
        # 时间卷积（膨胀卷积门控）
        # ============================================
        self.filter_conv = nn.Conv2d(
            residual_channels, dilation_channels,
            kernel_size=(1, kernel_size), dilation=dilation
        )
        self.gate_conv = nn.Conv2d(
            residual_channels, dilation_channels,
            kernel_size=(1, kernel_size), dilation=dilation
        )
        
        # ============================================
        # 空间卷积：GCN
        # ============================================
        self.nconv = nconv()

        # ---- 动态调制网络（逐层独立，感知不同尺度） ----
        # ===== 调制网络：仅浅层和中层使用 =====
        if layer_idx <= 7:  # 浅层 
            self.use_modulation = True
            self.temporal_attn = TemporalAttentionPooling(dilation_channels)
            self.mod_net = BilateralModulation(
                num_nodes, dilation_channels, 
                hidden_dim=mod_hidden_dim, 
                top_k=top_k
            )
        else:  # 深层 
            self.use_modulation = False

        # ============================================
        # 归一化层
        # ============================================
        self.bn = nn.BatchNorm2d(residual_channels)
        
        # ============================================
        # 输出头
        # ============================================
        self.output_conv = nn.Conv2d(
            residual_channels, out_dim, kernel_size=(1, 1)
        )
        # ============================================
        # Skip connection
        # ============================================
        self.skip_conv = nn.Conv2d(
            residual_channels, skip_channels, kernel_size=(1, 1)
        )

        
        
    def forward(self, x, base_adj):
        """
        Args:
            x: [B, C, N, T] 输入特征
        Returns:
            layer_output: [B, out_dim, N, T] 当前层的输出
            residual: [B, C, N, T] 残差（用于下一层）
        """
        residual_input = x
        
        # ============================================
        # 1. 时间卷积（门控膨胀卷积）
        # ============================================
        filter_x = torch.tanh(self.filter_conv(x))
        gate_x = torch.sigmoid(self.gate_conv(x))
        x_tcn = filter_x * gate_x

        s = self.skip_conv(x_tcn)
        
        # ============================================
        # Step 2: 计算动态调制图
        # ============================================
        # 2a. 时间注意力池化
        # x_tcn: [B, dilation_channels, N, T_tcn]
        # → x_pooled: [B, N, dilation_channels]
        
        # ===== 计算邻接矩阵 =====
        if self.use_modulation:
            # 浅层：输入感知动态调制
            x_pooled, _ = self.temporal_attn(x_tcn)
            A_use = self.mod_net(x_pooled, base_adj)
        else:
            # 深层：直接使用基础图
            B = x_tcn.size(0)
            A_use = F.softmax(base_adj, dim=-1).unsqueeze(0).expand(B, -1, -1)
        
        # ============================================
        # 2. 空间卷积（动态图卷积）
        # ============================================
       
        out = [x_tcn]
        
        # 一阶扩散
        x1 = self.nconv(x_tcn, A_use)
        out.append(x1)
        
        # 高阶扩散
        for k in range(2, self.order + 1):
            x2 = self.nconv(x1, A_use)
            out.append(x2)
            x1 = x2
        
        # 拼接所有阶
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        x_gcn = F.dropout(h, self.dropout, training=self.training)
        

        # ============================================
        # 3. 残差连接 + 归一化
        # ============================================
        T_gcn = x_gcn.size(3)
        x_out = x_gcn + residual_input[:, :, :, -T_gcn:]
        x_out = self.bn(x_out)

       
        return x_out, s





class DSTGNet(nn.Module):
    """双流独立解耦 + NAPL 空间嵌入"""
    def __init__(self, device, num_nodes, dropout=0.3, supports=None,
                 in_dim=2, out_dim=12, residual_channels=32,
                 dilation_channels=32, skip_channels=256, end_channels=512,
                 kernel_size=2, blocks=4, layers=2):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.total_layers = blocks * layers
        self.out_dim = out_dim
        
        
        self.static_supports = supports
        
        # ===== 1. 初始信号分解 =====
        self.decomposer = SignalDecompositionLayer(in_features=1)
        
        # ===== 2. 输入投影 =====
        self.start_conv_steady = nn.Conv2d(3, residual_channels, kernel_size=(1, 1))
        self.start_conv_dynamic = nn.Conv2d(3, residual_channels, kernel_size=(1, 1))
        
        # ===== 3. NAPL 空间嵌入 =====
        self.spatial_embed_steady = SpatialEmbedding(num_nodes, residual_channels)
        self.spatial_embed_dynamic = SpatialEmbedding(num_nodes, residual_channels)
     
        self.time_slot_memory = TimeSlotMemory(
            num_nodes=num_nodes, 
            residual_channels=residual_channels,
            steps_per_day=288  # 5分钟间隔，一天288个时间步
        )

        #自适应的动态邻接矩阵
        m, p, n = torch.svd(supports[0])
        initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
        initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
        self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)




        # ===== 4. 双流独立层 =====
        self.steady_blocks = nn.ModuleList()
        self.dynamic_blocks = nn.ModuleList()
        
        receptive_field = 1
        total_layer_idx = 0
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for l in range(layers):
                self.steady_blocks.append(
                    SteadyStreamBlock(
                        num_nodes, total_layer_idx, supports, residual_channels, dilation_channels,
                        skip_channels, out_dim, dropout, 
                        kernel_size, new_dilation
                    )
                )
                self.dynamic_blocks.append(
                    DynamicStreamBlock(
                        num_nodes, total_layer_idx, residual_channels, dilation_channels,
                        skip_channels, out_dim, dropout, kernel_size, new_dilation,
                        top_k=20,                   # 稀疏化：保留最强20条边
                        mod_hidden_dim=32           # 调制网络隐藏维度
                    )
                )
                total_layer_idx += 1
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
        
        self.receptive_field = receptive_field

        self.alpha = nn.Parameter(torch.tensor(0.6))
        
        # ===== 5. 最终输出聚合层 =====
        self.skip_proj_s = nn.Conv2d(skip_channels, end_channels, kernel_size=1)
        self.skip_proj_d = nn.Conv2d(skip_channels, end_channels, kernel_size=1)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1,1),
                                    bias=True)

        self.end_conv_3 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        

        
    def forward(self, input):
        B, C, N, T = input.shape
        
        
        # input: [B, C, N, T]  C=2: [流量, hod] 或 C=3: [流量, hod, dow]
        # 取 hod 通道（第1维的第2个通道）
        if C >= 2:
            hod = input[:, 1, :, :]  # [B, N, T]
            hod_indices = (hod * 287).long()  # [B, N, T]
        else:
            hod_indices = None
        # 取时间索引的平均值（所有节点的 hod 相同，取第一个节点即可）
        if hod_indices is not None:
            hod_indices_t = hod_indices[:, 0, :]  # [B, T]
        else:
            hod_indices_t = None

        # # ===== 1. 初始信号分解 =====
        x_for_decomp = input.permute(0, 3, 2, 1)
        if x_for_decomp.size(-1) == 2:
            x_for_decomp = torch.cat([x_for_decomp, torch.zeros(B, T, N, 1, device=input.device)], dim=-1)
        
        steady_x, dynamic_x = self.decomposer(x_for_decomp)
        steady_x = steady_x.permute(0, 3, 2, 1)
        dynamic_x = dynamic_x.permute(0, 3, 2, 1)
        
        # 输入投影
        x_steady = self.start_conv_steady(steady_x)
        x_dynamic = self.start_conv_dynamic(dynamic_x)
        
        # x_steady = self.start_conv_steady(input)
        # x_dynamic = self.start_conv_dynamic(input)
        # ===== 2. 新增：加入 NAPL 空间嵌入 =====
        x_steady = self.spatial_embed_steady(x_steady)
        x_dynamic = self.spatial_embed_dynamic(x_dynamic)
        
        # x_steady: [B, C', N, T]
        if hod_indices_t is not None:
            x_steady = self.time_slot_memory(x_steady, hod_indices_t)

        # 填充
        in_len = x_steady.size(3)
        if in_len < self.receptive_field:
            pad_len = self.receptive_field - in_len
            x_steady = F.pad(x_steady, (pad_len, 0, 0, 0))
            x_dynamic = F.pad(x_dynamic, (pad_len, 0, 0, 0))
        
        # ===== 3. 双流独立解耦 =====
        current_steady = x_steady
        current_dynamic = x_dynamic
        skip_s = 0
        skip_d = 0
    

        #自适应的动态邻接矩阵构建
        base_adj = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

        
        for steady_block, dynamic_block in zip(self.steady_blocks, self.dynamic_blocks):
            current_steady, s_s= steady_block(current_steady, self.static_supports)
            current_dynamic, s_d= dynamic_block(current_dynamic, base_adj)
            
            try:
                skip_s = skip_s[:, :, :, -s_s.size(3):]
                skip_d = skip_d[:, :, :, -s_d.size(3):]
            except:
                skip_s = 0
                skip_d = 0
            skip_s = skip_s + s_s
            skip_d = skip_d + s_d
        skip_s = self.skip_proj_s(skip_s)
        skip_d = self.skip_proj_d(skip_d)

        alpha_gate = torch.sigmoid(self.alpha)
        skip = (1 - alpha_gate) * skip_s + alpha_gate * skip_d

        # skip = skip_d
        x = F.relu(skip)

        x = F.relu(self.end_conv_2(x))
        x = self.end_conv_3(x)
        
        
        return x

class TimeSlotMemory(nn.Module):
    def __init__(self, num_nodes, residual_channels, steps_per_day=288):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(num_nodes, steps_per_day, residual_channels) * 0.1)
        # 可学习的融合权重：初始化为 0.5，让模型自己决定信多少
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x_steady, hod_indices):
        B, C, N, T = x_steady.shape
        hod_flat = hod_indices.reshape(B * T)
        mem = self.memory[:, hod_flat]  # [N, B*T, C]
        mem = mem.permute(1, 0, 2).reshape(B, T, N, C).permute(0, 3, 2, 1)
        
        # 可学习权重：训练初期 ~0.5，训练后可能收敛到 0.3~0.8
        alpha_gate = torch.sigmoid(self.alpha)
        return (1 - alpha_gate) * x_steady + alpha_gate * mem

