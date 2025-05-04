#UDIS-ShipLock

## TODO

- [x] 重新搞个BackBone，还是先H再Mesh但不用ResNet，重新封装个类
- [x] 把模型定义和计算结构搞成2个文件分开来
- [x] 看论文了解注意力机制一般加在哪里？加几个?
- [x] 确定新数据集训练时的图片大小，是和UDIS-D一样用512x512还是resize到保留原始纵横比?做2组实验对比
- [x] 重新整理数据集，删掉视差太大的
- [x] 整理仓库中文件夹结构
- [ ] 整理UDIS2在UDIS-Ship数据集的指标
- [ ] 训练UANet（DSELBlock/FPN/可变形卷积）

## 开发中遇到的问题

1.如何在主干网络加注意力机制？

目前的主干网络是ResNet，在它里面加注意力机制其实有2个思路：

- 加在残差块里面（也就是`resnet.py`里的`BasicBlock`类或者`BottleNeck`类）
- 加在残差块之间

ResNet其实不同变种的基本结构类似，包含的几个大层都是固定的，只是里面的残差块的数目不同

```python
def _forward_impl(self, x: Tensor) -> Tensor:
    # See note [TorchScript super()]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)

    return x

```

因此，可以把注意力机制加在几个大层(layer1，layer2...)之间

> 由于使用方法二的计算量更少，且可以使用预训练权重，所以采用方法二

注意：修改代码时，可以直接改`resnet.py`，也可以修改UDIS2的模型定义，我选择的是后者

参考链接：

- [pytorch中加入注意力机制（CBAM），以ResNet为例。解析到底要不要用ImageNet预训练？如何加预训练参数？ - 知乎](https://zhuanlan.zhihu.com/p/99261200)
- [【深度学习】Pytorch：在 ResNet 中加入注意力机制_注意力机制resnet-CSDN博客](https://blog.csdn.net/2303_80346267/article/details/145226698)
- [HAM:Hybrid attention module神经网络中混合注意力模块代码全网首次开源复现 - 知乎](https://zhuanlan.zhihu.com/p/555252748)



2.注意力机制要加几个呢？



3.到底可不可以在全连接层前面加全局池化？现在梯度爆炸/消失的问题到底出在残差块还是BN层还是全局池化

> 最好不要用全局池化到1x1，会丢失很多局部空间细节信息，CCL输出的是feature flow，每个位置其实都有用，所以不要池化到1x1

- [x] 加了Residual Block（内含BN层）就会出现nan
- [x] 不用残差块 + 全局池化到1x1再全连接————没问题
- [x] 不用残差块，但加上BN层 + 全局池化到1x1再全连接————没问题
- [x] 残差块但是去掉BN层————没问题

感觉见鬼了，现在加了残差块和全局池化到1x1也不会出现NAN了...先看看多训练会儿Loss能不能降得比较低——训了1h，Loss还是0.8几，通用iter数别的模型的Loss基本都0.4几了

- [ ] 试一下残差块 + 卷积下采样，不用全局池化，Loss下降的会不会快一些，因为2个论文都没用最大池化



4.换成yolo的backbone后，卷积的超参数怎么调？

- 直接和yaml中的一致，相对于yolo11-x，太占显存了，所以调小了些

[一文弄懂 YOLOv8网络结构解读 、yolov8.yaml配置文件详细解读与说明、模型训练参数详细解析 通俗易懂！入门必看系列！_yolov8细节信息-CSDN博客](https://blog.csdn.net/m0_74823452/article/details/145491780)



5.SPPF（空间金字塔池化）有啥用？

用于卷积和全连接层之间的连接，直接用全局池化损失的信息太多了，利用池化金字塔可以获得不同尺度的信息，减小信息丢失。

加了之后确实效果比之前直接用全局最大池化好一些

[空间金字塔池化改进 SPP / SPPF / SimSPPF / ASPP / RFB / SPPCSPC / SPPFCSPC-CSDN博客](https://blog.csdn.net/weixin_43694096/article/details/126354660)



6.回归网络里加不加BN层？

对于Low-Level的视觉任务最好别加，因为BN层会改变数据的分布，而回归网络需要预测准确的偏移，可能会受影响

> 各种Norm的区别？



7.怎么理解self-attention

首先这个模块虽然是NLP领域提出的，但是它的输出还是普通的Tensor，所以CV任务也能直接用。加入输入包含5个token，self-attention就是拿各个token分别与其他token计算一个相似度分数，然后就得到了输入中的各个token的重要程度（一个权重），这就是自注意力机制。

由于self-attention计算出的结果与位置无关，为了能使结果包含位置信息，通常会对输入加个位置编码



8.MultiHead Self-Attention和Single Head Self-Attention的区别

MultiHead中，一个Token会生成多组qkv，而Single Head Self-Attention中只生成一组



9.YOLO中的C3k2之类的模块到底用什么作用？

- [【YOLO】深入理解 CSP 瓶颈模块的变种：Bottleneck、C3、C3k、C2f 和 C3k2_csp模块-CSDN博客](https://blog.csdn.net/weixin_43694096/article/details/144034254)
- [深入浅出之C2F模块(YOLO)-CSDN博客](https://blog.csdn.net/a8039974/article/details/143116935)
- [YOLO全系列模块详解（持续更新）_c3k2-CSDN博客](https://blog.csdn.net/Alex_Tlover/article/details/142764338)



10.模型轻量化的方法：

- [深入浅出理解深度可分离卷积（Depthwise Separable Convolution）_dwconv-CSDN博客](https://blog.csdn.net/m0_37605642/article/details/134174749)
- [FasterNet（CVPR 2023）论文解读_fasternet论文-CSDN博客](https://blog.csdn.net/ooooocj/article/details/135574234)



11.`fine tuning.py`的代码输入到模型推理的时候需要resize到512，但是最后拼接的时候不会reszie，以原图大小进行拼接
