"""Analytical weight bytes per decode token for our precision and the canonical upstream builds.

Estimates only: active parameters x bytes per weight (BFP8 1088 B, BFP4 576 B per 1024 values),
ignoring padding, KV reads and activations. Compares against nominal GDDR6 bandwidth.

    python3 scripts/weight_traffic.py
"""
H=2816; L=30; Ls=25; Lg=5
attn_s = H*16*256 + 2*H*8*256 + 16*256*H
attn_g = H*16*512 + H*2*512 + 16*512*H        # K=V on global layers
attn = Ls*attn_s + Lg*attn_g
mlp = L*3*H*2112
exp_act = L*8*3*H*704
exp_all = L*128*3*H*704
router = L*H*128
lm = 262144*H
B = {"bf16":2.0, "bfp8":1088/1024, "bfp4":576/1024}
print(f"params active: attn {attn/1e9:.3f}B mlp {mlp/1e9:.3f}B experts {exp_act/1e9:.3f}B router {router/1e9:.4f}B lm {lm/1e9:.3f}B total {(attn+mlp+exp_act+router+lm)/1e9:.2f}B")
print(f"experts total {exp_all/1e9:.2f}B params")
def cfg(name, e, d, head, r="bf16"):
    tot = exp_act*B[e] + (attn+mlp)*B[d] + lm*B[head] + router*B[r]
    print(f"{name}: {tot/1e9:.2f} GB/token (experts {exp_act*B[e]/1e9:.2f}, dense {(attn+mlp)*B[d]/1e9:.2f}, head {lm*B[head]/1e9:.2f}); all-expert residency {exp_all*B[e]/1e9:.1f} GB")
    return tot
ours = cfg("ours (bfp4 exp, bfp8 dense+head)", "bfp4","bfp8","bfp8")
qb2 = cfg("canonical QB2 (bfp8 exp, bf16 dense+head)", "bfp8","bf16","bf16")
t3k = cfg("canonical T3K (bf16 all, assumed)", "bf16","bf16","bf16")
for n,tot,bw,ms in [("ours N300 dev 15.63ms",ours,576,15.63),("ours vLLM 18.72ms",ours,576,18.72),("QB2 gate 1000/45.69",qb2,2048,1000/45.69),("T3K demo 1000/19.9",t3k,8*288,1000/19.9)]:
    print(f"{n}: floor {tot/bw/1e6:.2f} ms at nominal {bw} GB/s; effective {tot/ms/1e6:.0f} GB/s = {tot/ms/1e6/bw*100:.0f}% of nominal")
