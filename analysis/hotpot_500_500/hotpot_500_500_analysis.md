# HotpotQA 500 bridge + 500 comparison 分析

- 样本：bridge 500，comparison 500，共 1000。
- k 范围：1–15；Oracle 并列时选择较小 k。
- combined 最优 fixed-k：k=8，平均 F1=0.203800。

| group | majority ratio | entropy (bits) | mean best-k | median | std |
|---|---:|---:|---:|---:|---:|
| bridge | 0.4840 | 2.8062 | 3.7360 | 2.00 | 3.8810 |
| comparison | 0.2740 | 3.2452 | 4.2560 | 3.00 | 3.7297 |
| combined | 0.3790 | 3.0745 | 3.9960 | 2.00 | 3.8150 |
