# 循环式神经算子：高频波场的多尺度代理建模

> **导读**：要提高模型的预测精度，除了增加参数，能不能让同一组参数多算几轮？循环式语言模型在文本生成中探索这个问题；高频波场的建模也面临相似的选择。本文把论文中的**自复合神经算子**（self-composing neural operator，SC-NO）称为**循环式神经算子**（Looped Neural Operator）［1］，用这个名字突出同一个模块沿计算深度反复调用的结构。

## 从 AI4S Cup 到循环式神经算子

这项工作起源于 2024 年 AI4S Cup 的「超声 CT 成像中的声场预测」赛题。比赛在 3 月 25 日至 5 月 31 日举行，要求根据给定的声速分布和声源预测超声波场。这是成像中的**正演**，不是直接从波形重建声速。我和何俊材、许波组成的队伍以包含循环更新的神经算子方案获得了这个赛题的第一名（一等奖）。赛题与评测入口见 [Bohrium 赛事页面](https://www.bohrium.com/en/competitions/2512153120?tab=introduce)，获奖名单见[主办方公告](https://maimai.cn/article/detail?efid=Z4u-2yCfrR1XSdC_pErEPQ&fid=1843859548)。

比赛之后，我们把参赛方案中的共享循环发展成更一般的神经算子结构，并研究如何从少轮开始训练、逐步增加循环深度［1］。感谢深势科技、北京科学智能研究院、北京大学和中国科学院声学研究所组织这场有趣的比赛，让不同团队能在同一个来自超声成像的计算问题上试验和交流各自的方法。

## 高频波场为何难以预测

下图前两列分别是声速分布和数值求得的参考波场。两幅图描述的是不同的物理量：前者告诉我们声波在各处传播得多快，后者告诉我们在给定声源与频率下，声压怎样随位置变化。同一频率下，声速不同，局部波长也不同；来自不同路径的波相互叠加，形成图中密集的干涉条纹。因此，声速图上某一处的变化，可能通过传播影响到相距很远的波场。

以一段声速为 1,500 米/秒的均匀介质为例，300 kHz 时的波长是 5 毫米，500 kHz 时只有 3 毫米。同样 1 毫米的偏移，分别相当于一个振荡周期的五分之一和三分之一。这不是实验的分辨率指标，而是一个量级估算：频率越高，只画对波场的大体轮廓越不够，还要把波峰和波谷放到正确的位置。

![声速分布、参考波场、预测波场与逐点误差](https://xlliu2017.github.io/images/wechat-20261115/academic/usct_forward_case.png)

**图 1｜高频超声正演样例。** 前三列依次为输入声速分布、参考波场和 MgNO-self 的预测波场；最右列为预测减参考，使用单独的色标。图中 Residual 是预测误差，不是把预测代回原方程后得到的残差。沿用报告图，见文献［1］图 5(a)。

传统正演是给定一组介质与声源，就求解一次相应的波动方程。**神经算子**学习的则是一族问题的解映射：输入是整幅介质参数场和声源，输出是整幅波场；换一个介质或声源，就得到另一幅波场预测。例如在反演中，介质估计每更新一次，就需要用新的声速图重新预测探头响应。若这些查询与训练覆盖的问题相近，代理模型有机会减少反复求解的成本。这里节省的是训练完成后的计算；获得训练数据和训练网络本身也有成本。

在传统的高波数 Helmholtz 求解中，怎样协调局部计算与全局传播也是核心问题。源转移区域分解法，把源对其他区域的影响以等效源逐层传递［2］。此外重叠子域求解改进粗层平滑的多层预条件，以及把全局粗层求解与局部子域求解结合的两层 Schwarz 预条件［3，4］。这些是求解离散方程的数值方法。本文的代理模型关注的是一个相关但不同的问题：能否在学习的模块里组织跨尺度的信息传递，让它直接预测波场？

## 多重网格负责跨尺度，循环结构负责多轮更新

第一种结构是**跨尺度的信息传递**。多重网格在不同分辨率上处理同一个问题。细网格保留局部变化，粗网格让相距较远的信息能在更少的网格点之间传递，再把修正送回细网格。MgNO 将这种粗细网格之间的运算用于网络中线性算子的参数化［5］。这不是说可以在粗网格上直接求完整的高频波场：传到粗层的是用于更新的特征，最终仍在细网格上输出波场。

怎样表示尺度差异，已有几条不同路线。傅里叶神经算子（FNO）在频域组织全局信息传递［6］。**谱偏置**指的是，在许多训练设置下，神经网络倾向于先拟合低频、变化较缓的成分，较晚才拟合高频、变化较快的成分。［7］从傅里叶角度研究了这个现象，称之为**频率原则**（Frequency Principle）。这里说的是训练的先后与难易，并不是说傅里叶表示不能容纳高频。我和合作者以分层注意力结合导数损失，让训练看到函数值之外的局部变化［8］；多尺度傅里叶神经算子（MscaleFNO）让不同分支处理经过尺度变换的输入，以适应振荡函数和散射问题［9］。这些工作提供了理解高频难点的不同视角，并非都参加了本文的同数据对照。

第二种结构是**重复使用更新规则**。循环式神经算子将同一个骨干模块连续应用多次。每一轮的输入包括前一轮的内部状态，以及保持不变的介质和声源信息；因此，同样的计算规则作用在不断变化的状态上。超声模型使用一个独立的初始化块，之后复用共享的多重网格块，最后将内部状态转为预测波场。我们把这类神经算子叫做自复合神经算子或者循环神经算子［1］。

![共享多重网格模块依次更新状态，并在每轮保留相同的介质和声源输入](https://xlliu2017.github.io/images/wechat-20261115/academic/shared_backbone.png)

**图 2｜循环式神经算子的信息流。** 蓝色横向箭头传递当前状态；上方红色虚线表示每轮都使用同一组介质与声源信息。图中多个 G 表示同一骨干模块的不同调用，不代表不同参数组。图示聚焦共享部分，未展开初始化块。取自 15 分钟报告的自复合结构页。

可以把图中的一次更新写成：

$$
h_{j+1} = G_{\theta}(h_j; c, f).
$$

这里 $c$ 是声速，$f$ 是声源，$h_j$ 是第 $j$ 次调用前积累的内部状态，$\theta$ 是共享参数。虽然每次调用的都是同一个 $G_{\theta}$，输入状态却不同：后一次可以在前一次已经提取的传播特征上接着修正。与把多个不同模块串起来相比，**多调用一轮不增加共享骨干的参数，但仍要付出一轮计算**。

这里借鉴的是「反复更新」的结构，而不是复刻一个数值求解器。比如北京大学团队参与的 Meta-MgNet 用网络从问题参数生成适配的平滑器，再放进多重网格求解过程［10］；循环式神经算子则更新网络内部状态，直接输出预测场。内部状态趋于稳定不等于原方程残差趋于零；训练规定深度下的精度也不能自动推广到任意多轮调用。

## 与 Looped Transformer 的联系：循环发生在计算深度上

通常的自回归大型语言模型（LLM）每生成一个词元（token，即分词后的单位），就把它接到已有文本后面，再预测下一个。虽然每步调用同一个模型，**输入序列却在增长**。循环式 Transformer（Looped Transformer）增加的是另一种循环：在预测下一个词元时，可以让同一个 Transformer 核心反复更新**同一段上下文**的内部表征，再读出词元。增加内部循环轮数不必增加输出文字的长度；它和通过生成更多中间文字来延长显式思维链，是两种不同的计算分配方式。

较早的 Universal Transformer 已经研究了让注意力模块沿深度反复使用［11］。近年的循环深度语言模型进一步考察：保持权重不变，能不能通过增加内部循环来改善部分推理任务［12］？Ouro 还在预训练中学习怎样分配循环深度［13］。这些工作说明，除了增加参数量和生成长度，模型还可以通过改变**内部更新次数**来分配推理计算。

回看图 2，MgNO-self 也把「有多少参数」与「用这些参数算多少轮」分开。但它的共享核心是**多重网格块**而不是 Transformer，更新的是空间网格上的波场特征，不是词元表征；它在固定介质、声源和频率下预测整幅波场。多走一轮既不是生成一个词元，也不是把声波向物理时间的下一刻推进。两类模型相似的是计算结构，不是输入、输出或训练任务。

神经算子领域也在探索由粗到细的多轮预测。例如，迭代细化神经算子（IRNO）从预训练的算子预测出发，另加一个反复使用的细化模块，并在训练中逐轮加强对高频误差的关注［14］。相比之下，MgNO-self 把多重网格共享模块放在预测器内部，随模型一起训练。两者的共同问题是，增加的每一轮计算能否带来有用的修正；不能仅凭网络有循环结构就预期误差一定更小。

## 渐进加深如何影响训练

参数共享让训练可以先从较浅的模型开始，再逐级加深。渐进展开训练（Train-and-Unroll）在 400 kHz 的一个较小超声实验中依次使用深度 **3、4、5**，每阶段训练 10 轮，并以上一阶段的权重初始化下一阶段。对照方案从一开始就用目标深度 5，同样训练 30 轮。两种方案最终得到的模型深度相同，但渐进方案前 20 轮处理的模块调用更少。在论文报告的一次 NVIDIA RTX A6000 计时中，渐进方案用时 **23.96 小时**，直接训练用时 **32.04 小时**，训练时间减少约 25%。这不是推理加速比，也不意味着更换骨干和数据后会得到同等收益［1，第 3 节］。

![渐进加深与直接训练的损失曲线](https://xlliu2017.github.io/images/wechat-20261115/academic/training_strategy_comparison_guarded_lr3e4.png)

**图 3｜两种训练方式的损失曲线。** 红线为渐进加深，蓝线为直接训练。虚线标出红线所对应的两次深度切换；横轴是优化步数，不是墙钟时间。沿用报告图，见文献［1］图 3(a)。

读图时，可以将虚线前后的红线看成三个阶段：切换深度时，损失是在不同调用次数下计算的，短时波动不等于训练失效。与蓝线比较时，最终阶段才是同一目标深度下的损失对照；比较计算成本，则要看单独记录的实际用时。

这里的「逐步增加循环」发生在**训练**中，每个阶段仍更新模型权重；它不等于固定权重后，在**推理时**按样本难度增加计算。本篇的波场结果按选定深度报告。若要像循环式语言模型那样按问题难度分配计算，还要在同一组权重下比较不同深度的精度与实际用时，并检验怎样判断某个样本可以少算一轮。

## 从单个波场到测试集

在基于 OpenBreastUS 的主实验中，研究使用 7,200 个训练样本，并在 900 个留出的测试样本上比较三档频率的波场预测［1，15］。这一由国内多家机构参与建立的基准，通过数值模拟提供配对的声速图和波场。下表使用**相对均方根误差**：将预测和参考波场的差异按参考波场的大小归一化。数值越小，表示整体预测越接近参考，但一个全场平均指标不能告诉我们误差具体集中在哪里，这也正是图 1 的右列值得看的原因。

| 频率 | FNO | AFNO | BFNO | MgNO-self |
| --- | ---: | ---: | ---: | ---: |
| 300 kHz | 2.69% | 1.65% | 1.13% | **0.28%** |
| 400 kHz | 4.26% | 2.42% | 1.48% | **0.36%** |
| 500 kHz | 4.90% | 2.76% | 2.09% | **0.49%** |

*表 1｜从论文表 2 选列，数值换算为百分数。FNO 即傅里叶神经算子；AFNO、BFNO 为论文采用的两种对照模型。*

500 kHz 时，MgNO-self 的误差约为 FNO 的十分之一、BFNO 的四分之一。这比较的是**完整模型**。在另外一组范围较小的 400 kHz 骨干对照中，MgNO 自复合前后的验证误差为 1.94% 和 1.51%，而 FNO 与 U-Net 的自复合版本在该短训练设置下均未改善［1，表 3］。因此，实验支持的是一种与骨干设计相配合的建模方法，不是反复调用任意网络都能提高精度。

这组测试回答的是：在所选采集设置与声速分布下，方法对留出样本的预测如何。若边界形状本身变化，就要把几何如何输入模型、怎样验证新几何另列为问题。北京大学、清华大学等团队参与的点云神经算子（PCNO）从离散表示入手，将几何及定义在其上的场作为点云处理［16］；它关注的问题与这里的计算复用不同，不能把两边的实验结论互相外推。

训练后的 MgNO-self 按预设深度输出预测，并不以原方程残差为停止条件。在需要大量同类波场预测时，这种结构可以成为有价值的代理；在要求达到指定离散方程容限的场合，则仍需单独验算。这也引出了另一条路线：让神经网络参与迭代校正，由求解器保留最终的停止判据。

## 参考文献

1. He J, Liu X, Xu J. [Self-composing neural operators for high-frequency and multiscale PDE surrogates](https://doi.org/10.1016/j.jcp.2026.115189). *Journal of Computational Physics*, 2026, 565: 115189.
2. Chen Z, Xiang X. [A Source Transfer Domain Decomposition Method for Helmholtz Equations in Unbounded Domain](https://doi.org/10.1137/130917144). *SIAM Journal on Numerical Analysis*, 2013, 51(4): 2331–2356.
3. Lu P, Xu X. [A Robust Multilevel Preconditioner Based on a Domain Decomposition Method for the Helmholtz Equation](https://doi.org/10.1007/s10915-019-01015-z). *Journal of Scientific Computing*, 2019, 81(1): 291–311.
4. Lu P, Xu X, Zheng B, et al. [Two-Level Hybrid Schwarz Preconditioners for the Helmholtz Equation with High Wave Number](https://doi.org/10.1137/24M168533X). *SIAM Journal on Numerical Analysis*, 2025, 63(6): 2187–2220.
5. He J, Liu X, Xu J. [MgNO: Efficient Parameterization of Linear Operators via Multigrid](https://arxiv.org/abs/2310.19809). *International Conference on Learning Representations (ICLR)*, 2024.
6. Li Z, Kovachki N, Azizzadenesheli K, et al. [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895). *International Conference on Learning Representations (ICLR)*, 2021.
7. Xu ZQJ, Zhang Y, Luo T, et al. [Frequency Principle: Fourier Analysis Sheds Light on Deep Neural Networks](https://doi.org/10.4208/cicp.OA-2020-0085). *Communications in Computational Physics*, 2020, 28(5): 1746–1767.
8. Liu X, Xu B, Cao S, et al. [Mitigating spectral bias for the multiscale operator learning](https://doi.org/10.1016/j.jcp.2024.112944). *Journal of Computational Physics*, 2024, 506: 112944.
9. You Z, Xu Z, Cai W. [MscaleFNO: Multi-scale Fourier neural operator learning for oscillatory functions and wave scattering problems](https://doi.org/10.1016/j.jcp.2025.114530). *Journal of Computational Physics*, 2026, 547: 114530.
10. Chen Y, Dong B, Xu J. [Meta-MgNet: Meta multigrid networks for solving parameterized partial differential equations](https://doi.org/10.1016/j.jcp.2022.110996). *Journal of Computational Physics*, 2022, 455: 110996.
11. Dehghani M, Gouws S, Vinyals O, et al. [Universal Transformers](https://arxiv.org/abs/1807.03819). *International Conference on Learning Representations (ICLR)*, 2019.
12. Geiping J, McLeish S, Jain N, et al. [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3b01972cf31e6fa0fe29e4b8b5c2a0a1-Abstract-Conference.html). *Advances in Neural Information Processing Systems (NeurIPS)*, 2025, 38: 46242–46293.
13. Zhu RJ, Wang Z, Hua K, et al. [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741). *arXiv*, 2025: 2510.25741.
14. Liu X, Shang S, Wang X, et al. [Iterative Refinement Neural Operators are Learned Fixed-Point Solvers: A Principled Approach to Spectral Bias Mitigation](https://proceedings.mlr.press/v306/liu26dm.html). *International Conference on Machine Learning (ICML)*, 2026, PMLR 306: 78216–78247.
15. Zeng Z, Zheng Y, Hu H, et al. [OpenBreastUS: Benchmarking Neural Operators for Wave Imaging Using Breast Ultrasound Computed Tomography](https://arxiv.org/abs/2507.15035). *arXiv*, 2025: 2507.15035.
16. Zeng C, Zhang Y, Zhou J, et al. [Point cloud neural operator for parametric PDEs on complex and variable geometries](https://doi.org/10.1016/j.cma.2025.118022). *Computer Methods in Applied Mechanics and Engineering*, 2025, 443: 118022.
