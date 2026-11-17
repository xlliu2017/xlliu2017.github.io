# 从声波测量到声速图：生成模型与物理约束下的超声断层成像

> **导读**：生成模型能画出很逼真的图像，但科学成像的问题是：这幅图能不能解释仪器实际测到的信号？在超声断层成像中，测量不完整时，数据往往不足以确定所有细节；反复计算波场又会拖慢重建。Diff-ANO［1］用生成先验提出候选声速图，再用波动方程检查并修正它。

## 探头记录的是波，而不是声速

超声断层成像（USCT）可以用一圈换能器来采集数据：一个位置发射，其他位置接收；更换发射位置，得到新的一组记录。把每次发射得到的接收信号排成一列，便得到图 1 右侧的矩阵。图中的数值不是某个组织像素的声速；**重建的声速图**才描述物体内部每个位置的传播速度，单位可用米/秒，而非图像灰度。

仅看波首次到达的时间，可以判断某些传播路径整体较快还是较慢；但当介质里有明显的折射、散射和波的叠加，单个到时很难说明这些变化发生在哪里。全波形反演将更多记录纳入比较。在本文的频域设置中，给定频率下的复数读数同时携带振幅与相位。Sandhu 等人研究了如何用环阵数据进行频域波形反演［2］；Zhang 等人则用源编码减少重复的波场计算，并以互相关调整反演的更新方向［3］。OpenBreastUS 提供了供正演与反演比较使用的波动模拟数据［4］。

![环形换能器及以声源为列、接收器为行的观测矩阵](https://xlliu2017.github.io/images/wechat-20261115/academic/Physics_illustration.png)

**图 1｜超声断层成像的测量方式。** 左图是环形发射和接收位置及散射波的示意；右图显示一组完整观测矩阵的实部，两个坐标轴是发射位置与接收位置的索引，不是组织内的空间坐标。沿用报告和文献［1］图 1。

## 两种信息：本次测量与生成先验

设 $c$ 是待求的声速图，$A(c)$ 是预测的探头读数。计算 $A(c)$ 时，先按声速图和各发射位置求波场，再在实际接收位置读取相应信号。测量记为 $y = A(c) + \varepsilon$，其中 $\varepsilon$ 表示噪声。**给定声速计算探头信号是正问题，给定信号寻找声速是反问题。** 环上探头间距增大，是在多个角度上减少采样；探头只覆盖一段弧，则还缺少整段观测方向。在两种情况下，都可能出现不同声速分布产生相近读数。

贝叶斯关系可写成：**后验正比于似然乘以先验**。

$$
p(c \mid y) \propto p(y \mid c)\,p(c).
$$

这里的似然是「假如这幅声速图成立，看到当前读数有多合理」。例如，如果各个实、虚部分量含有独立、等方差的高斯噪声，那么负对数似然（去掉常数）就是预测与观测的平方误差除以两倍噪声方差；本文用于校正的是读数的平方误差［1］。先验表达的是在看这一次读数之前，哪些声速结构比较合理。它可以通过显式的正则项给出，例如用全变差惩罚约束声速图的空间变化［5］；也可利用不经样本预训练的网络结构作隐式约束，在单个测量上优化［6］。Diff-ANO 选择从已有声速样本中学习生成先验。

这里最重要的区分是：**样本先验不能代替本次测量提供的信息；测量无法区分的细节，也不能仅凭生成图像逼真就认作已证实。** 上述贝叶斯关系解释了两种信息的角色，并不意味着 Diff-ANO 能精确抽取后验样本或给出已校准的不确定性。

## 先由生成模型提出估计，再用波动方程校正

Diff-ANO 首先用直接反演网络从读数估计一幅初步声速图。它不是最终重建，而是引导生成模型保留这次测量已能辨识的结构。在**一致性模型**中，同一条生成轨迹上处于不同噪声水平的状态，学习映射到相近的干净图像估计，因此可以从较高噪声水平直接得到一幅候选图，不必逐小步走完整条去噪轨迹［1，7］。这里说的是算法设定的噪声水平，与仪器测量噪声不是一回事。

生成候选与物理校正交替进行：每次得到候选图，都用其预测探头读数，再根据预测与实测之间的偏差调整声速。这与直接反演网络只输出一次不同，也不同于把一个无条件扩散模型移植到波动反演后，在大量去噪小步中分别做测量引导。

![Diff-ANO 原始框架图：粗重建、条件一致性模型和神经伴随优化](https://xlliu2017.github.io/images/wechat-20261115/academic/Framework.png)

**图 2｜Diff-ANO 重建流程。** 从左上角读起，测量先经过直接反演得到粗图，再通过条件分支引导一致性模型；最右侧的神经伴随优化提供数据校正，红色虚线标出多步过程。沿用报告和文献［1］图 2。

为什么需要**伴随计算**？如果逐个像素试探改变声速会让预测信号怎样变化，每次改动都要重新求波场。伴随方法反过来把接收端的误差作为驱动，将其传播回计算区域；再与原来的正演波场结合，就能为整幅声速图计算校正方向，不用逐像素重求。即便如此，常规反演每更新一次声速，仍要按新声速为多组发射重新计算正演和伴随波场。这才是主要的计算瓶颈之一。

论文利用特定采集与声学设置下的线性叠加，把伴随场写成以接收位置为源的正演响应的加权和，权重由读数差给出；这样可以复用成批计算的波场，不再单独求解每个伴随问题。这批波场又由多重网格神经算子（MgNO）近似，校正方向按波动模型导出的伴随公式构造［1，第 4 节］。因此，减少成本有两层：少做独立的方程求解，每次需要的波场又算得更快。相应的代价是，近似波场产生的校正方向也可能有误差。

## 重建质量：比较同一组模拟测量

图 3 展示了稀疏视角、5 dB 信噪比下的四个模拟样本。可以先看第二行左列，认清外部轮廓与内部小结构，再比较直接反演网络（InvNet）和最右列的 Diff-ANO 是否把这些结构放在相应的位置。只看轮廓容易忽略局部偏差，只看图像锐利程度也可能把不属于参考图的纹理当成恢复出来的细节。论文因此报告峰值信噪比（PSNR）与结构相似性（SSIM）；二者分别反映与参考声速图在数值和结构上的接近程度，不是临床诊断指标。

![四个稀疏视角模拟样本的参考声速图及不同算法重建结果](https://xlliu2017.github.io/images/wechat-20261115/academic/exp_sparse_recon.png)

**图 3｜Sparse-I 模拟重建。** 从左到右为参考声速图、传统迭代及学习方法的重建，最右列 Ours 为 Diff-ANO。图中每行是一个样本，不是测试集均值。沿用报告和文献［1］图 7。

为了区分两部分的作用，可以看论文中的两组对照。其一是**加不加物理校正**：在另一组更稀疏的 Sparse-II 设置、同为 5 dB 时，只用条件一致性模型得到 26.18 dB、0.8901；加入三步神经伴随校正后，平均值变为 27.91 dB、0.9231［1，表 1］。这反映的是平均改善，不代表每个样本每一步都会变好。

其二是**用什么办法算校正所需的波场**：保持初步反演模块不变，在 Sparse-I 中用 MgNO-II 时，PSNR 是 32.07 dB，SSIM 是 0.9732；换回收敛 Born 级数（CBS）数值求解器是 32.85 dB 和 0.9803［1，表 7］。这里的 CBS 是同一重建框架内部的波场对照，不是图 3 中独立运行的 CBS 迭代反演。用代理模型能省时间，也可能牺牲一部分重建精度；只用网络自己的波场误差无法衡量这一取舍。

## 计算效率：把在线重建和离线训练分开

以下是论文报告的 Sparse-I 仿真比较［1，表 3］。DPS（扩散后验采样）原本是针对一般带噪反问题的生成方法［8］。**表中的 DPS 是本文将它适配到超声反演的实现**：把原框架中的测量算子换成声学方程与接收采样，并用它计算测量引导。DDS 是论文比较的另一种扩散采样方法。

| 重建方法 | 生成网络调用次数 | 成批波场计算次数 | 平均单样本在线用时 |
| --- | ---: | ---: | ---: |
| CBS 迭代反演（GPU） | 0 | 60+ | 298.5 秒 |
| DPS | 1,000 | 1,000 | 4,023.7 秒 |
| DDS | 50 | 100+ | 451.9 秒 |
| Diff-ANO | 5 | 5 | **1.1 秒** |

表中一次波场计算指对所有源做一批预测，不是只算一个源。DPS 已用 GPU 上成批运行的数值方法计算波场，但需要 1,000 轮；Diff-ANO 用一致性模型减少生成轮数，每次波场计算又换成神经算子。因此，表格比较的是整条在线重建流程，不是某一个求解模块在相同容限下的加速倍数。

这项比较不含训练。仅波场代理 MgNO 的训练就约需 130 个 NVIDIA A100 GPU 小时，生成模型的训练开销另计。摊销这些成本需要考虑模型能为多少同类测量重复使用，以及更换探头布局或介质分布后是否需要额外训练。论文还在 OpenPros 声速图上重新模拟波形以考察不同数据条件［9］，但没有直接用临床采集的波形验证。**这里的秒级结果说明的是数值反演流程的计算潜力，不是临床系统的计时或诊断结论。**

## 从流匹配到生成式成像：快与可信需要分别检验

生成模型的快速发展提供了不同的先验。除扩散和一致性模型外，**流匹配**（flow matching）学习一个随算法时间变化的更新方向，把简单的噪声分布变换为图像分布。FlowDPS 将测量的引导加入这个生成过程，用于超分辨率、去模糊等线性反问题［10］。这个联系也说明了超声反演的额外开销：去模糊时可以用已知的模糊算子检查候选图；超声里，一旦换了一张候选声速图，波动方程的系数也随之变化，通常要为多个发射位置重新计算波场。换一个更快的生成模型，不会自动消除这笔费用。

与超声更接近的是地震成像。RED-DiffEq 将预训练扩散模型用作地下速度反演的正则化项，同时保留波动方程给出的数据误差［11］。两类应用都是从记录的波形推断介质，但生成先验进入算法的位置不同：RED-DiffEq 用它约束迭代中的速度模型；Diff-ANO 用条件一致性模型提出候选图，再用神经算子减少波场校正的计算。地震实验给出了可以比较的研究思路，并不为超声实验的速度或精度提供直接证据。

生成式成像还面对一个独立于速度的问题：**符合观测不等于所有细节都由观测确定。** 线性断层成像已有研究分析，失准的先验怎样在测量难以分辨的部分引入虚假结构［12］。这里的声速反演是非线性的，不能直接套用那项线性分析，但应追问同样的问题：在不同候选图都能解释有限探头读数时，哪些局部差异还有额外证据？后续可用未纳入反演的发射和接收通道检验预测信号，并在已知结构的实体模体上检查小结构有没有被漏掉或凭空出现。这些是需要另做的验证，不是本文已有的结果。

## 参考文献

1. Cao X, Ding Q, Liu X, et al. [Diff-ANO: Towards Fast High-Resolution Ultrasound Computed Tomography via Conditional Consistency Models and Adjoint Neural Operators](https://arxiv.org/html/2507.16344v2). *arXiv*, 2026: 2507.16344v2.
2. Sandhu GY, Li C, Roy O, et al. [Frequency domain ultrasound waveform tomography: breast imaging using a ring transducer](https://doi.org/10.1088/0031-9155/60/14/5381). *Physics in Medicine & Biology*, 2015, 60(14): 5381–5398.
3. Zhang N, Zhao Y, Yuan Y, et al. [Cross-correlation adjustment full-waveform inversion with source encoding in ultrasound computed tomography](https://doi.org/10.1016/j.ultras.2024.107392). *Ultrasonics*, 2024, 142: 107392.
4. Zeng Z, Zheng Y, Hu H, et al. [OpenBreastUS: Benchmarking Neural Operators for Wave Imaging Using Breast Ultrasound Computed Tomography](https://arxiv.org/abs/2507.15035). *arXiv*, 2025: 2507.15035.
5. Wang K, Matthews T, Anis F, et al. [Waveform inversion with source encoding for breast sound speed reconstruction in ultrasound computed tomography](https://doi.org/10.1109/TUFFC.2014.006788). *IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control*, 2015, 62(3): 475–493.
6. Yan W, Zhang Q, Wu Y, et al. [A Plug-and-Play Untrained Neural Network for Full Waveform Inversion in Reconstructing Sound Speed Images of Ultrasound Computed Tomography](https://arxiv.org/abs/2406.08523). *arXiv*, 2024: 2406.08523.
7. Song Y, Dhariwal P, Chen M, et al. [Consistency Models](https://proceedings.mlr.press/v202/song23a.html). *International Conference on Machine Learning (ICML)*, 2023, PMLR 202: 32211–32252.
8. Chung H, Kim J, McCann MT, et al. [Diffusion Posterior Sampling for General Noisy Inverse Problems](https://arxiv.org/abs/2209.14687). *International Conference on Learning Representations (ICLR)*, 2023.
9. Wang H, Wu Y, Feng Y, et al. [OpenPros: A Large-Scale Dataset for Limited View Prostate Ultrasound Computed Tomography](https://openreview.net/forum?id=kcFEpBagea). *International Conference on Learning Representations (ICLR)*, 2026.
10. Kim J, Kim BS, Ye JC. [FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems](https://openaccess.thecvf.com/content/ICCV2025/html/Kim_FlowDPS__Flow-Driven_Posterior_Sampling_for_Inverse_Problems_ICCV_2025_paper.html). *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2025: 12328–12337.
11. Shan S, Zhu M, Lin Y, et al. [Regularization by denoising diffusion models for solving inverse PDE problems with application to full waveform inversion](https://doi.org/10.1038/s42005-026-02677-6). *Communications Physics*, 2026, 9: 242.
12. Bhadra S, Kelkar VA, Brooks FJ, et al. [On Hallucinations in Tomographic Image Reconstruction](https://doi.org/10.1109/TMI.2021.3077857). *IEEE Transactions on Medical Imaging*, 2021, 40(11): 3249–3260.
