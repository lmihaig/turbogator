# Breakdown of ezgatr steps

x = (B,T,C,D)

- B = batch_size
- T = sequence length?
- C = input channels
- D = vector dimensions

I think it's staying hardcoded (8,T,2,16)

# MVOnlyGATrModel(B, T, C, D)

```
y = MVOnlyGATrEmbedding(B,T,C,D)
for each block
    y = MVOnlyGATrBlock(y)
EquiLinear(y)
```
