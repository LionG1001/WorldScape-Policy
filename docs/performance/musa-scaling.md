# MUSA 扩展效率优化记录

本轮拆出可选ZeRO-2 shard padding、公共MUSA环境配置和监控同步优化。单机不限制MCCL channels；双机保留默认上限14，可显式覆盖。padding默认关闭，避免将实验结果扩大为通用默认。

## 条件与口径

1×8或2×8 S5000，80GiB/卡；torch/torch_musa 2.11.0.post2、MUSA5.2、MCCL2.4、原生MUSA DeepSpeed0.19.3。BF16、ZeRO-2、每卡batch8、累积1，单机GBS64、双机GBS128。RoboCOIN折衣数据，native模拟EEF适配、stride1、3视角160×320、33帧、动作96×20。50步、warmup0.05，关闭保存和W&B。

统一使用第11–50步端到端均值，包含数据等待和监控，不剔除慢步；全50步另见CSV。扩展效率=T单/T双，吞吐加速比=2×效率。不同rank不是独立重复实验。

|节点|channels上限|padding bytes|workers/rank|稳态均值s|P90 s|
|---|---:|---:|---:|---:|---:|
|单机|14|0|4|6.535755|6.737953|
|单机|默认（实际28）|0|4|5.693712|6.022859|
|单机|默认（实际28）|128|4|5.639202|7.279983|
|双机|14|0|4|6.667670|6.975373|
|双机|14|128|4|6.690219|9.760224|
|双机|32|0|4|6.579723|8.171353|
|双机|32|128|4|6.963333|10.644908|
|双机|32|128|8|5.233234|5.637506|

以上均完成所有rank的50个optimizer step，正常exit0，loss/梯度有限且采样参数更新通过。双机workers8的全50步均值6.905252s，启动至退出631.27s，峰值allocated33.69–34.34GiB/卡，loss首末0.9059246→0.4542787。

## 归因与边界

- 单机14channel上限拖慢optimizer通信；移除上限后的均值恢复至5.693712s。原生SDPA仍实际命中flash_musa。
- 实际大小的16rank BF16 all-gather微基准：14channels未对齐2.015820s→128字节对齐0.069802s；32channels为0.889763s→0.039373s。微基准不代表整步等比例加速。
- padding后workers4数据供应出现长尾，双机整步没有改善，这一负结果保留。仅workers4→8，同32channels/pad128均值6.963333→5.233234s，降低24.85%；每步最慢rank的DataLoader.next均值2.339813→0.100853s。
- 单机workers8对照和双机14channels/workers8仍在测试。不能拿workers4较慢的单机基准宣称98%达标；当前PR提供已验证机制，最终最佳channels和扩展效率仍待配对复测。
- 8/16rank小模型对齐前后3步loss和所有参数逐位一致；16rank相同alignment checkpoint API保存/加载通过。完整模型恢复、恢复后下一步一致性和长期收敛未认证。
- 组合定向测试31项通过；CPU staging补充修正后ZeRO定向3项通过。保存alignment元数据，恢复时不匹配会拒绝加载。

## 复现配置

在已准备好数据和模型的环境，单机使用专用脚本，双机在每台分别设置NODE_RANK、MASTER_ADDR和本机MCCL_IB_HCA后启动多机脚本。公共入口继承这些变量，无需改安装目录。

```bash
export RUN_STEPS=50 PER_DEVICE_TRAIN_BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=1
export MAX_STEPS=50000 OPTIMIZER_WARMUP_RATIO=0.05 DATASET_STEP_STRIDE=1
export SAVE_STEPS=0 SAVE_AT_END=false DATALOADER_NUM_WORKERS=8
export WSP_MUSA_ATTENTION_BACKEND=sdpa WSP_MUSA_ZERO_ALIGNMENT_BYTES=128
# 单机：不要设置MCCL_MIN_NCHANNELS/MCCL_MAX_NCHANNELS
bash recipes/posttrain/posttrain_agilex_fold_shirt_text_musa.sh
# 双机：每台执行，设置各自NODE_RANK、HCA以及共同MASTER_ADDR
MCCL_MAX_NCHANNELS=32 bash recipes/posttrain/posttrain_agilex_fold_shirt_text_musa_multinode.sh
```

默认monitor用于正确性/显存检查；上述性能来自额外同步的端到端计时回调，不能将普通控制台内部step_time直接当成同口径。公开记录不包含机器地址、客户文档、原始日志或内部依赖源码。
