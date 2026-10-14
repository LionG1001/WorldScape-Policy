# MUSA attention 适配与性能记录

本轮基于 `88df179`，修复长度、缩放、局部窗口和空序列处理，并增加可选 MUSA FlashAttention varlen 路径。默认仍为原生 SDPA；不依赖外部 FlashAttention 才能训练。

## 算子对照

S5000，torch/torch_musa 2.11.0.post2、MUSA 5.2，BF16，B=8/H=24/D=128；相同输入，前向+反向，5次预热、30次测量。外部 FlashAttention 2.8.3.post1 为 MUSA 构建。

|Q/K长度|原生SDPA ms|外部FA2 ms|
|---|---:|---:|
|1/1|0.218|0.326|
|50/50|0.224|0.340|
|100/150|0.320|0.383|
|100/475|0.423|0.517|
|24/475|0.387|0.521|
|1000/512|1.202|1.320|
|1000/257|1.019|1.139|

最大前向/梯度绝对差为0–0.001953。Trace确认原生SDPA实际使用flash_musa前向与反向；API名SDPA不代表math fallback。上述形状外部FA2较慢，因此不默认启用。短序列随机梯度首次反传曾遇到原生muDNN workspace错误；最终对照两路径均先以sum().backward预热，不表示底层首次调用问题已修复。

## 训练与验证范围

最新适配代码在单机8卡、双机16卡各完成50步；每卡batch8、累积1、BF16 ZeRO-2、原生模拟EEF数据适配器、stride1、warmup0.05，关闭checkpoint和W&B。固定第11–50步端到端均值，同14channels基线单机6.535755s、双机6.667670s。该训练同时包含后续监控优化，不能把整步变化归因于attention修正。单机14channels会限制通信性能，也不能用这组效率宣称扩展优化达标。

定向测试覆盖CPU语义及7项MUSA外部FA前向/反向场景；包含attention、monitor、ZeRO、runtime和数据适配的组合测试31项通过。完整模型外部FA50步训练、CUDA数值一致性及长期收敛尚未认证。外部依赖源码与内部构建补丁不包含在此PR中。
