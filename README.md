# Q-VGM / LIBERO-Spatial

- [当前 idea 改进实验](idea改进实验.md)
- [阶段总结与未决问题（交接入口）](阶段总结与未决问题.md)
- [复现记录、数学原理和文件用法](复现过程.md)
- [数据 pipeline：offline 与 online](数据PIPELINE.md)
- [论文算法逐项核对](论文算法核对.md)
- [当前主线：原文算法与发布 checkpoint 输入](configs/libero_spatial_checkpoint.yaml)
- [输入边界核对配置](configs/libero_spatial_paper.yaml)（先行采集失败，已停止扩大）
- [旧实验配置与共享路径](configs/libero_spatial.yaml)

当前实现是独立 OpenPI/LIBERO offline 流程，引用现有 few-shot SFT 权重；不依赖 RLinf trainer。实际完成情况以复现记录和运行产物为准，online 尚未实现。

当前主线已完成：52/500=10.4%，未复现提升。原复现核查已收尾；2026-09-24 用户授权另开 idea 改进分支，详见当前实验文档。
