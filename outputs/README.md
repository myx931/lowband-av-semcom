# Outputs

实验输出使用 `outputs/{experiment_name}/{timestamp}/`。除本说明外，本目录内容均被 Git 忽略。

每次正式实验应保存：

- 完整解析后的配置；
- Git commit hash 和随机种子；
- Python、PyTorch、CUDA 与设备环境；
- 训练日志和最佳模型路径；
- 评价结果 JSON；
- 示例重建视频路径。

模型权重、视频和日志只保存在本地或经批准的外部存储中，不提交到仓库。
