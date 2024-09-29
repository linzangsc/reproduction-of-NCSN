# 【学习笔记】深度生成模型（八）：Fisher散度、Denoising Score Matching和Score-based Model

## 写在前面

本文是对人大高瓴人工智能学院 李崇轩教授主讲的公开课第七节得分函数模型部分内容的梳理（课程链接[了解Sora背后的原理，你需要学习深度生成模型这门课！ 人大高瓴人工智能学院李崇轩讲授深度生成模型之原理与应用（第1讲）_哔哩哔哩_bilibili](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1yq421A7ig/%3Fspm_id_from%3D333.337.search-card.all.click%26vd_source%3D196c5d43f645df8f93e712dc5e152b18)）。如有不准确的地方还请大家指出。

本文梳理的是得分函数模型Score-based Model(SBM)，并且将会沿着EBM的likelihood-free的学习方法逐渐过渡到SBM，因此会用到大量EBM中的基础知识，推荐和上一篇一起食用。

回顾一下EBM对似然的定义：

$$\begin{align}
p_{\theta}(x) &= \frac{\exp(f_{\theta}(x))}{Z_{}(\theta)} \tag{1} \\
&= \frac{\exp(f_{\theta}(x))}{\int \exp(f_{\theta}(x))dx} \tag{2} \\
\end{align}$$

在上一篇梳理能量函数模型EBM时，我们已经借Langevin Dynamics(LD)引入了得分函数score function的定义。回顾一下score function和LD：

$$\begin{align}
s_{\theta}(x) &:= \nabla_x \log p_{\theta}(x) \tag{3} \\
x_{t+1} &= x_t + \epsilon s_{\theta}(x_t) + \sqrt{2 \epsilon} z_t, z_t \sim \mathcal{N}(0, 1) \tag{4} \\
\end{align}$$

对于EBM而言，其score function的形式十分简单。将式（1）代入式（3）得到EBM的score function就是神经网络的输出$$f_{\theta}(x)$$对样本$$x$$的梯度：

$$\begin{align}
s_{\theta}(x) &= \nabla_x f_{\theta}(x)  - \nabla_x \log Z(\theta) = \nabla_x f_{\theta}(x) \tag{5} \\
\end{align}$$

其中$$Z(\theta)$$关于$$x$$是一个常量，与$$x$$无关。我们可以通过MLE结合LD来驱动EBM的学习，这里面会用到Contrastive Divergence的手段，但问题在于CD对似然函数的梯度的估计是一种有偏的估计，而引入CD的根本原因是EBM的似然是intractable的。

既然对EBM做MLE是存在问题的，且EBM的score function是非常好算的，那么有没有办法从score function出发驱动EBM的学习？答案是我们可以不进行MLE，而是通过最小化Fisher散度的方法实现EBM。

## Fisher Divergence和Denoising Score Matching

Fisher散度是直接从score function出发定义的一个散度：

$$\begin{align}
D_F(p_{data} \Vert p_{\theta}) = \frac{1}{2}\mathbb{E}_{x \sim p_{data}}[\Vert \nabla_x \log p_{data}(x) - s_{\theta}(x)\Vert_2] \tag{6} \\
\end{align}$$

Fisher散度就是数据分布$$p_{data}(x)$$和模型分布$$p_{\theta}(x)$$的score function的二范数，再对数据分布$$p_{data}$$取期望。将式（5）代入式（6），得到

$$\begin{align}
D_F(p_{data}\Vert p_{\theta}) = \frac{1}{2}\mathbb{E}_{x \sim p_{data}}[\Vert \nabla_x \log p_{data}(x) - \nabla_x f_{\theta}(x)\Vert_2] \tag{7} \\
\end{align}$$

由于取了二范数，因此Fisher散度是大于等于0的，且可以证明的是当$$D_F(p_{data}\Vert p_{\theta})$$等于0时，等价于模型分布$$p_{\theta}(x)$$就等于数据分布$$p_{data}(x)$$。我们令式（6）等于0，则$$f_{\theta}(x)$$和$$\log p_{data}(x)$$在函数空间上梯度处处相等，则它们最多相差一个常数$$C$$：

$$\begin{align}
f_{\theta}(x) = \log p_{data}(x) +C \tag{8} \\
\end{align}$$

将式（8）代入EBM建模的概率分布式（2）：

$$\begin{align}
p_{\theta}(x) &= \frac{\exp(f_{\theta}(x))}{\int \exp(f_{\theta}(x))dx} \tag{9} \\
&= \frac{\exp(\log p_{data}(x) +C)}{\int \exp(\log p_{data}(x) +C) dx} \tag{10} \\
&= \frac{\exp(C)\exp(\log p_{data}(x))}{\exp(C)\int \exp(\log p_{data}(x)) dx} \tag{11} \\
&= \frac{p_{data}(x)}{\int  p_{data}(x) dx} \tag{12} \\
&= p_{data}(x) \\
\end{align}$$

因此当$$D_F(p_{data}\Vert p_{\theta})$$等于0时，EBM的模型分布就等于数据分布。所以我们可以不做MLE，转而通过最小化Fisher散度来更新EBM，这样的方法称为Score Matching：

$$\begin{align}
\min_{\theta} D_F(p_{data}\Vert p_{\theta}) = \frac{1}{2}\mathbb{E}_{x \sim p_{data}}[\Vert \nabla_x \log p_{data}(x) - \nabla_x f_{\theta}(x)\Vert_2] \tag{13} \\
\end{align}$$

式（13）实际上还是没法直接用，原因在于数据分布的score function是未知的，好在Fisher散度可以写成不带$$\nabla_x \log p_{data}(x)$$的形式：

$$\begin{align}
\min_{\theta} D_F(p_{data}, p_{\theta}) &= \mathbb{E}_{x \sim p_{data}}[\frac{1}{2}\Vert \nabla_x f_{\theta}(x)\Vert_2 + tr(\nabla_x^{2}f_{\theta}(x))] \tag{14} \\
&= \frac{1}{n} \sum_{i=1}^n [\frac{1}{2}\Vert \nabla_x f_{\theta}(x_i)\Vert_2 + tr(\nabla_x^{2}f_{\theta}(x_i))] \tag{15} \\
\end{align}$$

式（14）到式（15）是使用了蒙特卡罗估计。通过最小化式（15）来驱动EBM更新，然后在训练完成以后再通过郎之万动力学采样，是EBM的一种likelihood-free的学习方法。和基于CD的学习方法相比，score matching不需要在每个训练轮次都多次采样，但不足之处是$$f_{\theta}(x_i)$$的二阶导也就是hessian矩阵的计算也是复杂度比较高的，从这个角度来说score matching是很难scale up的。

为了能使Score Matching的计算效率进一步提升，Vincent提出了Denoising Score Matching(DSM)。DSM提出，如果我们往$$x$$中加高斯噪声$$z \sim \mathcal{N}(0, \mathbb{I})$$，得到$$\tilde{x} = x + \sigma z$$，则加噪后的$$\tilde{x}$$的真实分布$$q(\tilde{x})$$和模型分布$$p_{\theta}(\tilde{x})$$之间的Fisher散度等价于：

$$\begin{align}
\min_{\theta} D_F(q, p_{\theta}) &= \mathbb{E}_{\tilde{x} \sim q(\tilde{x})}[\frac{1}{2}\Vert s_{\theta}(\tilde{x}) - \nabla_{\tilde{x}} \log q(\tilde{x}) \Vert_2]  \tag{16} \\
&= \mathbb{E}_{x \sim p_{data}, \tilde{x} \sim q(\tilde{x}|x)}[\frac{1}{2}\Vert s_{\theta}(\tilde{x}) - \nabla_{\tilde{x}} \log q(\tilde{x}|x) \Vert_2] + \C  \tag{17} \\
\end{align}$$

其中，$$\C$$是常量，我们在后面将省略；$$q(\tilde{x}|x)$$表示加噪后的变量$$\tilde{x}$$关于$$x$$的条件分布，且由于$$\tilde{x} = x + \sigma z$$，在给定$$x$$的条件下$$q(\tilde{x}|x) \sim \mathcal{N}(x, \sigma^2 \mathbb{I})$$，因此我们可以写出这个条件分布的概率密度函数：

$$\begin{align}
q(\tilde{x}|x) = \frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(\tilde{x} - x)^2}{2\sigma^2}) \tag{18} \\
\end{align}$$

将式（18）代入式（17），得到：

$$\begin{align}
\min_{\theta} D_F(q, p_{\theta}) &= \mathbb{E}_{x \sim p_{data}, \tilde{x} \sim q(\tilde{x}|x)}[\frac{1}{2}\Vert s_{\theta}(\tilde{x}) - \nabla_{\tilde{x}} \log q(\tilde{x}|x) \Vert_2]  \tag{19} \\
&= \mathbb{E}_{x \sim p_{data}, \tilde{x} \sim q(\tilde{x}|x)}[\frac{1}{2}\Vert s_{\theta}(\tilde{x}) - \nabla_{\tilde{x}} \log \frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(\tilde{x} - x)^2}{2\sigma^2}) \Vert_2]  \tag{20} \\
&= \mathbb{E}_{x \sim p_{data}, \tilde{x} \sim q(\tilde{x}|x)}[\frac{1}{2}\Vert s_{\theta}(\tilde{x}) + \frac{\tilde{x} - x}{\sigma^2}  \Vert_2]  \tag{21} \\
&= \mathbb{E}_{x \sim p_{data}, z \sim \mathcal{N(0, \mathbb{I})}}[\frac{1}{2}\Vert s_{\theta}(x + \sigma z) + \frac{z}{\sigma}  \Vert_2]  \tag{22} \\
\end{align}$$

式（22）就是DSM的目标函数。我们从直觉上理解一下式（22）。我们知道score function指向的是使得样本的概率密度最大的方向，那么对于一个加噪的样本$$\tilde{x}$$，它的概率密度增长最大的方向就应该是去噪的方向，这个方向就是$$-\frac{\tilde{x} - x}{\sigma^2}$$，所以式（22）是希望神经网络尽可能地预测去噪的方向。值得一提的是，这里和diffusion model里预测噪声的目标是非常相似的，但不同之处在于这里预测的不是噪声$$z$$，而是去噪的方向。

DSM绕开了hessian矩阵的计算，但代价是通过DSM估计得到的分布并不是原始数据的分布$$p_{data}$$，而是先对原始数据加了一次噪声，然后估计带噪分布$$q$$。只有在噪声的方差足够小的情况下（$$\sigma$$足够小），才可以认为$$q \approx p_{data}$$。

到这里，我们回顾一下likelihood-free的EBM训练方法，我们需要的是score function（式22）；在EBM训练完以后，我们依然要通过LD采样，需要的也是score function（式4）。既然我们在学习和采样过程中，都只需要score function，那可不可以抛开概率密度函数$$p_{\theta}(x)$$和normalizing constant $$Z(\theta)$$，抛开能量函数$$f_{\theta}(x)$$，直接用神经网络对$$s_{\theta}$$建模？这就是SBM的motivation。

## Score-based Model

梯度场和概率密度函数$$p_{\theta}(x)$$是一体两面的，因为梯度指向的是使得当前输入的概率密度最大的方向。 得分函数模型Score-based Model

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
