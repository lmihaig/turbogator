# Breakdown of ezgatr steps

x = (B,T,C_in,D)

- B = batch_size
- T = sequence length?
- C_in = input channels
- D = vector dimensions

I think it will stay hardcoded (8,T,2,16) but keep it as variables for now

- C_hid = hidden channels = 32
- C_int = intermediate channels = 32
- C_out = output channels = 1
- H = attention heads = 4
- L = number of layers = 4

MVOnlyGATrModel.forward(x)
|
|-- torch.mean(x)  -> reference: [B, 1, 1, D]
|
|-- MVOnlyGATrEmbedding(x)
|   |-- EquiLinear(C_in, C_hid)  -> y: [B, T, C_hid, D]
|
|-- Loop L times:
|   |-- MVOnlyGATrBlock(y, reference)
|       |
|       |-- MVOnlyGATrAttention(y)
|       |   |-- residual = y
|       |   |-- EquiRMSNorm(y)
|       |   |-- EquiLinear(C_hid, 3 *H* C_hid)  -> qkv: [B, T, 3*H*C_hid, D]
|       |   |-- rearrange (0 FLOPs)               -> q, k, v: [B, H, T, C_hid, D]
|       |   |-- w.exp() for w in attn_mix         -> Exponentiate learnable scalars
|       |   |-- equi_geometric_attention(q, k, v) -> attn_out: [B, H, T, C_hid, D]
|       |   |-- rearrange (0 FLOPs)               -> attn_flat: [B, T, H*C_hid, D]
|       |   |-- EquiLinear(H * C_hid, C_hid)      -> proj: [B, T, C_hid, D]
|       |   |-- proj + residual                   -> Element-wise Add: [B, T, C_hid, D]
|       |                                         -> Returns attn_y: [B, T, C_hid, D]
|       |
|       |-- MVOnlyGATrMLP(attn_y, reference)
|           |-- residual = attn_y
|           |-- EquiRMSNorm(attn_y)
|           |
|           |-- MVOnlyGATrBilinear(attn_y, reference)
|           |   |-- EquiLinear(C_hid, 4 * C_int)  -> bil_out: [B, T, 4*C_int, D]
|           |   |-- torch.split (0 FLOPs)         -> lg, rg, lj, rj: [B, T, C_int, D]
|           |   |-- geometric_product(lg, rg)     -> gp_out: [B, T, C_int, D]
|           |   |-- equi_join(lj, rj, reference)  -> ej_out: [B, T, C_int, D]
|           |   |-- torch.cat (0 FLOPs)           -> cat_out: [B, T, 2*C_int, D]
|           |   |-- EquiLinear(2* C_int, C_hid)  -> Returns bil_y: [B, T, C_hid, D]
|           |
|           |-- scaler_gated_gelu(bil_y)          -> Element-wise: [B, T, C_hid, D]
|           |-- EquiLinear(C_hid, C_hid)          -> proj: [B, T, C_hid, D]
|           |-- proj + residual                   -> Element-wise Add: [B, T, C_hid, D]
|                                                 -> Returns block_y: [B, T, C_hid, D]
|
|-- EquiLinear(C_hid, C_out)                      -> out: [B, T, C_out, D]
