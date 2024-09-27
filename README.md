# reproduction-of-SBM
# 【学习笔记】深度生成模型（八）：Score-based Model

## 写在前面

本文是对人大高瓴人工智能学院 李崇轩教授主讲的公开课第六节能量函数模型部分内容的梳理（课程链接[了解Sora背后的原理，你需要学习深度生成模型这门课！ 人大高瓴人工智能学院李崇轩讲授深度生成模型之原理与应用（第1讲）_哔哩哔哩_bilibili](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1yq421A7ig/%3Fspm_id_from%3D333.337.search-card.all.click%26vd_source%3D196c5d43f645df8f93e712dc5e152b18)）。如有不准确的地方还请大家指出。

![image-20240927145903976](C:\Users\l00850616\AppData\Roaming\Typora\typora-user-images\image-20240927145903976.png)

既然EBM推导到最后，我们用score matching的方法能够进行优化，且score matching和采样都只用到了score function而与normalizing constant $$Z$$无关了，那么我们能不能抛弃概率密度或者似然，直接定义score function来学习一个生成模型？这就是score-based model的motivation。

## 网络架构

score-based model用一个神经网络接收样本$$x$$为输入，输入$$x$$的score function，即$$\nabla_x \log p(x)$$也就是$$x$$的梯度场。和基于概率的生成模型不同的是，score function是和输入同维度的。以图片为例，score-based model类似于接受一个图片输入，输出一张和输入同维的图片，而基于概率的生成模型例如EBM，输出则是一个表征输入能量的标量（然后通过归一化被转换为概率密度）。这也是为什么score-based model使用的网络架构是U-Net，因为U-Net在同样是image2image的任务上（例如分割）已经被证明是非常有效的网络架构。

## Pitfall

Pitfall指的是宋飏博士在提出score-based model时指出的score-based model的一个问题，为了说明这个问题，我们需要从score-based model的优化目标也就是Fisher散度出发：

$$\begin{align}
D_F(p_{data}\Vert p_{\theta}) &= \frac{1}{2}\mathbb{E}_{x \sim p_{data}}[\Vert \nabla_x \log p_{data}(x) - \nabla_x f_{\theta}(x)\Vert_2] \tag{22} \\
&=  \frac{1}{2}\int p_{data}(x) \Vert \nabla_x \log p_{data}(x) - \nabla_x f_{\theta}(x)\Vert_2 dx \tag{22} \\
\end{align}$$

![image-20240927154503535](C:\Users\l00850616\AppData\Roaming\Typora\typora-user-images\image-20240927154503535.png)

由于Fisher散度在得分函数的二范数之外还对数据分布$$p_{data}(x)$$求了期望，因此对于空间中概率密度很低的样本$$x$$（例如从标准高斯噪声中采样得到的一个样本，它和MNIST数据集里的图像长得很不一样，它在数据分布中的概率密度理论上应该接近0），这些样本的Fisher散度被优化到的概率极低，结果就是用于估计样本的score function的神经网络对空间中概率密度很低的样本点估计出来的梯度场是严重失准的。在这种情况下，如果我们按照langevin dynamics(LD)进行采样，从标准的高斯噪声中采样一个噪声然后计算它的score function，并通过多次梯度上升得到采样结果，那么往往并不会得到有意义的结果，原因在于模型对这样的样本估计的梯度方向或者score function是不准的。

有一个非常直觉的手段可以解决这个问题：加噪。假设我们要建模分布的数据集的真实分布是一个具有明显波峰的结构，在这些波峰对应的样本点是概率密度很高的点；现在我们对原始图像加一个标准高斯噪声，那么均值0对应的概率密度理论上将会变大。我们对加噪之后的图像估计score function，优化Fisher散度的过程中对于标准高斯采出来的样本的梯度估计就会更准，最终在运用LD进行采样时，从标准高斯随机采一个噪声出发进行梯度上升的梯度就会更准，也就更有可能顺着准确的梯度方向爬到加噪后的数据分布的峰值上。

然而，这样还是没有完全解决问题，因为这样一来，我们最终采出来的样本服从的分布将是加噪后的数据分布，是模糊的不清晰的。假如我们加噪加多了，加完噪图像自己成噪声了，那么我们采出来的也就是噪声；假如我们加噪加少了，从标准高斯中随机初始化的噪声估计的score function又没那么准了。好在我们可以通过多步加噪解决这个问题。想象一下，我们从一张清晰的原始图像$$x_0$$开始，按顺序一共要对它加10次噪，得到$${x_1,...,x_{10}}$$，每个噪声都是以0为均值的高斯噪声，加噪的幅度逐渐变大（高斯噪声的方差逐渐变大），全部加完以后原图就变成了一张高斯噪声。接着，我们有10个神经网络模型$$f_{\theta}^1,...,f_{\theta}^{10}$$，用于估计10个score function，每个score function都是加了噪声以后的图片的score function。然后我们通过最小化Fisher散度训练好了这10个模型，现在要进行采样了，我们从高斯分布中采样一个噪声作为LD的初值$$x_{10}$$，然后先用$$f_{\theta}^{10}(x_{10})$$得到$$x_{10} \to x_{9}$$的梯度场，并通过几步梯度上升得到$$x_9$$。这里我们估计的梯度是准确的，因为$$f_{\theta}^{10}$$估计的就是$$x_9$$服从的分布（一个接近高斯分布的分布）对应的score function，由于$$x_{10}$$就是从高斯分布里采的，所以它对应的概率密度一定是不低的，也就意味着$$f_{\theta}^{10}(x_{10})$$是对梯度方向的准确估计。在得到$$x_9$$以后，我们重复上述的采样过程，就像从山脚顺着$$f_{\theta}^i$$爬山的过程，爬山一共被分为了10个阶段，每个阶段的梯度方向都是准确的，最终到达清晰图像$$x_0$$对应的真实数据分布的概率密度的峰顶。为了简单，也可以通过参数共享，使用一个神经网络模型来同时估计10个数据分布的score function，在这种情况下就需要对输入额外加上表示当前时间步$$t$$的特征（例如positional embedding)。
