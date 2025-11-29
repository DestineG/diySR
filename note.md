## 项目

- 早停机制
- 复现论文(liif, lte)
- 模型参数信息 | 各阶段耗时信息(print 结合 > a.log 实现)
- augment
- test函数实现(使用exp_xx.hydra | 自定义config进行测试)

## 问题

- 研究scale较大时锯齿原因
- liif引入二阶 lte引入e^i 增强高频细节

## 模型优化方向

- 模型图像平滑缺陷(Over-smoothing)，方案：模型改进，损失函数改进
- 生成质量提高(加入语义引导)
- 模型轻量化，加速推理

## commit

- 修复verbose配置错位bug