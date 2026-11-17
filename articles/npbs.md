# 高频波动方程的学习型预条件：从 Born 级数到迭代校正

> **导读**：已有一个便宜的传统预条件，怎样在它的基础上学到更有效的校正？神经预条件 Born 级数（NPBS）保留经典方法中对残差的参考变换，让网络在变换后生成波场校正，并用同一个变换衡量校正效果。**传统预条件不只是实验中的对照，也决定了网络学什么、在每轮迭代中怎么用，以及训练时如何评价它。**

## 波场求解的困难在何处

下图左侧是一个地下断层模型。声速在断层两侧突变，点源发出的波在介质中传播、反射并相互叠加。在给定频率下，Helmholtz 方程求的是同时包含振幅与相位的复数波场；中间一列显示波场的实部。右侧记录了不同算法的**相对残差**：把当前波场代回原方程，衡量还剩多少不平衡。同一条水平线代表同一残差大小；曲线更早到达这条线，表示所需迭代更少。它不直接告诉我们用了多少秒。

![OpenFWI CurveFault-B 单个样本的声速、波场及四种求解方法的残差曲线](https://xlliu2017.github.io/images/wechat-20261115/academic/curvefaultb_convergence.png)

**图 1｜断层介质中的二维 Helmholtz 求解。** 左为声速模型，白色叉号标记源位置；中为波场实部；右为收敛 Born 级数（CBS）、NPBS，以及两者与柔性广义最小残差法（FGMRES）结合后的残差曲线。横轴是迭代数，纵轴为对数尺度，相邻数量级的变化看的是倍数而不是差值。沿用报告中的 NPBS 单样本图［1］，介质来自 OpenFWI 的 CurveFault-B 数据集［2］。正文表格另列公开论文报告的测试集均值。

求解高频波首先要区分两类误差。第一类来自**离散**：把连续方程搬到网格上后，如果每个波长内的采样不够，计算出来的波即使满足离散方程，也可能逐渐偏离正确的传播相位。高波数有限元的误差分析说明了这一问题［3］。第二类是**迭代误差**：网格、边界处理和离散格式都固定以后，当前迭代结果离离散方程的解还有多远。NPBS 改进的是第二步：在同一张网格上，用更有效的校正逼近离散解。它不能补救过粗网格带来的相位偏差。

在传统迭代求解中，预条件子的设计已经有多种做法。陈志明（Zhiming Chen）与合作者的源传递区域分解法将源的影响以等效源逐层传递，可用于预条件 GMRES［4］；Leng 与 Ju 则研究了加性重叠区域分解［5］。许学军（Xuejun Xu）与合作者分别用重叠子域求解替代粗层上的普通平滑，以及用局部正交分解构造全局粗层求解器，与局部求解组成两层 Schwarz 预条件［6，7］。

湘潭大学团队的傅里叶神经求解器和 Wave-ADR-NS 则从频域校正和不同类型误差的分工处理入手［8，9］。NPBS 的不同点是：**用经典 Born 参考算子同时规定网络在迭代中处理什么残差，以及训练用什么度量衡量校正。** 上述区域分解方法是相关的传统路线，不是下文计时表中的对照项。

## 以经典预条件为起点：一起设计校正和损失

用 $Au = f$ 表示离散方程，$u$ 是波场，$f$ 是声源，$A$ 编码介质与波动规律。将当前波场代入，$r = f - Au$ 是还没有被平衡的驱动项。如果校正量 $\Delta u$ 正好满足 $A\Delta u = r$，把它加到当前波场上就能得到离散解。但精确计算这个增量与原问题一样难。预条件的目的，是用更便宜的步骤估计一个有用的增量，反复减少剩余的不平衡。

收敛 Born 级数提供了一个可复用的步骤［10］：把残差当作声源，先求它在一个均匀、带人工阻尼的参考介质中的响应。记这个参考问题的逆算子为 $G_{\eta}$。原介质非均匀，计算它的响应很难；参考问题系数不随位置改变，在论文的网格实现中可以借助快速傅里叶变换求解。这里的阻尼是计算手段；对方程两边施加同一个可逆变换，得到 $G_{\eta} Au = G_{\eta} f$，解并没有变。

**改变每轮如何校正。** 给定残差 $r$，CBS 先计算参考响应 $G_{\eta}r$，再逐点乘以系数，得到波场增量。NPBS 保留第一步，将第二步换成依赖介质的神经算子。固定介质后，将网络记为 $M_{\theta}$，波场增量就是［1，第 4.1 节］：

$$
\Delta u = M_{\theta}(G_{\eta}r).
$$

传统算子不是用来初始化网络权重，也不是为网络提供要模仿的答案，而是每轮校正的一部分。

从方程看，原来要解的校正方程 $A\Delta u = r$，现在写成 $(G_{\eta} A)\Delta u = G_{\eta}r$。当 $A$ 可逆时，网络的理想目标也从直接近似 $A^{-1}$，变为近似预条件后的逆作用 $(G_{\eta} A)^{-1}$。所求的波场增量没有变；变的是把已知的参考传播交给经典计算，再让网络学习怎样利用这一响应。

**改变训练如何衡量校正。** 网络给出增量 $\Delta u$ 后，原方程的新残差是 $r_{\mathrm{new}} = r - A\Delta u$。如果训练仍以原残差衡量，就比较 $\lVert r_{\mathrm{new}}\rVert_2 / \lVert r\rVert_2$；但在 NPBS 的独立迭代中，下一轮网络实际收到的是 $G_{\eta}r_{\mathrm{new}}$。于是训练改为最小化以下比值的样本平均［1，第 4.2 节］：

$$
\frac{\lVert G_{\eta}r_{\mathrm{new}}\rVert_2}{\lVert G_{\eta}r\rVert_2}.
$$

这就是**度量匹配**：训练直接衡量下一轮输入的残差相对于本轮缩小了多少。两个设计相互配合，不必更换或加大网络骨干。

损失度量的是训练样本上的平均单步改善，不保证每个新问题的每一步都缩小残差。实际迭代仍用原方程相对残差 $\lVert f - Au\rVert_2 / \lVert f\rVert_2$ 验收，并没有放宽原方程的停止标准。这个判据可以直接检查离散方程的满足程度；在高波数病态系统中，同样的残差容限不等于同样的波场误差，更不是实际介质和连续模型的精度保证。

## 二维实验：学到的预条件比基准好吗？

在 50 个 CurveFault-B 测试样本上，原方程相对残差容限为 $10^{-6}$，即把不平衡量降低到声源向量范数的百万分之一。下表选列论文在 NVIDIA H100 上报告的平均结果；各学习配置均使用多重网格神经算子（MgNO）作为骨干［1，表 3 及附录 G、H］。

| 求解配置 | 平均迭代数 | 平均求解时间 |
| --- | ---: | ---: |
| CBS 独立迭代 | 1,124 | 0.086 秒 |
| NPBS（MgNO）独立迭代 | 123 | 0.107 秒 |
| FGMRES + 仅参考算子 $G_{\eta}$ | 894 | 2.821 秒 |
| FGMRES + CBS | 547 | 1.820 秒 |
| FGMRES + 直接训练的 MgNO | 47 | 0.116 秒 |
| FGMRES + NPBS（MgNO） | **37** | **0.078 秒** |

判断预条件是否更有效，先看**同一外层求解器**中的比较。FGMRES 用预条件子产生候选校正方向，再组合这些方向以减少原方程残差。在相同的 FGMRES 配置下，NPBS 的迭代数和平均在线用时均低于只用参考算子、用完整 CBS 校正或直接训练同一 MgNO 骨干的对照［1，表 3］。只保留传统参考变换并不足以得到这个结果，后面的学习校正也起作用。

把 CBS 和 NPBS 各自拿出来独立迭代，则是另一组比较。NPBS 将平均步数从 1,124 降到 123，但网络使每轮更贵，总时间反而由 0.086 秒增至 0.107 秒。因此，这里的「比传统基准更好」有明确条件：以相同容限比较，在上述 FGMRES 配置中既减少迭代也缩短在线用时；不能理解成独立迭代也一定更快。

看这类结果时，图 1 适合判断单个样本在哪个阶段收敛，表格才给出测试集及硬件上的时间比较。不同网络也有不同的计算成本：论文还比较了五种骨干，迭代最少的骨干并非用时最短的骨干［1，表 2］。因此，选哪种网络不能只看收敛曲线。最后要问的是：在**同样的停止标准**下，减少的迭代是否足以覆盖每轮增加的计算？

## 三维实验：把两项设计的贡献分开看

三维 Kimberlina-V1 实验保持 MgNO 骨干架构不变，分别训练三种配置，在 192³ 网格、50 个测试样本上比较［1，表 4 及附录 G］。192³ 对应约 708 万个网格点，不是在二维数据上多跑几次。这个对照只改变残差输入与训练方式，有助于判断改善来自哪一步。

| 三维 MgNO 的配置 | 平均迭代数 | 平均求解时间 |
| --- | ---: | ---: |
| 原残差输入，按原残差训练 | 445 | 39.1 秒 |
| 预条件输入，仍按原残差训练 | 68 | 5.69 秒 |
| 预条件输入，按匹配度量训练 | **41** | **3.60 秒** |

第一行到第二行保持原残差度量，改变的是网络所作用的残差表示，平均迭代由 445 步降到 68 步；第二行到第三行使用同样的参考变换，只替换损失度量，进一步降到 41 步。这个对照支持的是：**在相同骨干下，传统预条件改变了网络怎样参与求解，匹配的损失又让学到的校正更有效。** 从 39.1 秒到 3.60 秒是这三种学习配置的在线求解时间比较，说明训练后的模型用起来更快，而不是训练本身更快，也不是相对经典 CBS 的三维加速比。

NPBS 也不同于团队另一项多通道多重网格预条件研究 McMg［11］：后者关注如何在粗网格保留局部波的传播信息；这里的变量是 Born 参考变换下的网络输入与训练度量，实验设置和结果不混用。

## 与 PDE 基础模型的关系：预测和求解是两件事

科学计算中的**基础模型**试图把不同模拟任务的信息放进同一套预训练权重。例如，清华大学等团队的 DPOT 在多个偏微分方程（PDE）数据集上预训练，再适配不同的下游预测任务［12］。它关心的是一次训练得到的表示能否迁移到新任务。NPBS 关心的问题不同：给定这一次的离散方程和容限，网络能否帮助数值迭代更快达标？基础模型给出的波场可能是很好的近似，但是否达标，仍需把它代回这一次的方程检查。

近年的**学习型迭代求解**也在探索怎样保留这项检查。以 G-RANS 为例，它根据稀疏矩阵的连接关系和当前残差，生成一组校正方向，再通过子空间投影组合它们［13］。NPBS 不从矩阵图生成校正空间，而是先用参考介质的传播算子变换残差，再学习波场增量。二者的共同点是网络参与中间步骤，目标仍由原方程定义；不同点是选择学习求解器里的哪个操作。

基础模型与学习型迭代也可以配合：让预训练的代理给出初始波场，再用 NPBS 校正到指定残差。这是尚待检验的组合，不是本文的实验结果。要判断收益来自哪里，可以在同一组介质、声源和硬件上，以未改动的求解器为基线，比较「只改初值」「只改校正」「两者都改」，记录达到同一容限的比例和总用时，而不只看初始预测有多准。

论文还在对流—扩散—反应方程、非线性方程 Newton 迭代中的线性子问题上检验了同样的设计［1，第 5.2 节］。可复用的做法是先找一个计算便宜的参考逆算子，再用它一起设计网络的校正机制和训练损失。不同方程仍要选择各自的参考算子并训练校正网络；这些实验不代表某一组权重已经能够跨方程求解。

在实际使用中，还要区分只求一次与反复求解。同一类速度模型下如果要换许多发射点，训练成本可能由多次求解分摊；若只需计算一个新问题，立即可用的经典方法可能更合适。是否采用学习校正，需要在目标网格、硬件和介质分布上，将训练开销、在线用时和达标率一起比较。

## 参考文献

1. Wang J, Jia J, Liu X. [Born-Preconditioned Learning for Iterative Neural PDE Solvers](https://arxiv.org/abs/2603.18527). *Advances in Neural Information Processing Systems (NeurIPS)*, 2026 (已录用).
2. Deng C, Feng S, Wang H, et al. [OpenFWI: Large-scale Multi-structural Benchmark Datasets for Full Waveform Inversion](https://proceedings.neurips.cc/paper_files/paper/2022/hash/27d3ef263c7cb8d542c4f9815a49b69b-Abstract-Datasets_and_Benchmarks.html). *Advances in Neural Information Processing Systems (NeurIPS)*, 2022, 35: 6007–6020.
3. Li Y, Wu H. [FEM and CIP-FEM for Helmholtz Equation with High Wave Number and Perfectly Matched Layer Truncation](https://doi.org/10.1137/17M1140522). *SIAM Journal on Numerical Analysis*, 2019, 57(1): 96–126.
4. Chen Z, Xiang X. [A Source Transfer Domain Decomposition Method for Helmholtz Equations in Unbounded Domain](https://doi.org/10.1137/130917144). *SIAM Journal on Numerical Analysis*, 2013, 51(4): 2331–2356.
5. Leng W, Ju L. [An Additive Overlapping Domain Decomposition Method for the Helmholtz Equation](https://doi.org/10.1137/18M1196170). *SIAM Journal on Scientific Computing*, 2019, 41(2): A1252–A1277.
6. Lu P, Xu X. [A Robust Multilevel Preconditioner Based on a Domain Decomposition Method for the Helmholtz Equation](https://doi.org/10.1007/s10915-019-01015-z). *Journal of Scientific Computing*, 2019, 81(1): 291–311.
7. Lu P, Xu X, Zheng B, et al. [Two-Level Hybrid Schwarz Preconditioners for the Helmholtz Equation with High Wave Number](https://doi.org/10.1137/24M168533X). *SIAM Journal on Numerical Analysis*, 2025, 63(6): 2187–2220.
8. Cui C, Jiang K, Liu Y, et al. [Fourier Neural Solver for Large Sparse Linear Algebraic Systems](https://doi.org/10.3390/math10214014). *Mathematics*, 2022, 10(21): 4014.
9. Cui C, Jiang K, Shu S. [A Neural Multigrid Solver for Helmholtz Equations with High Wavenumber and Heterogeneous Media](https://doi.org/10.1137/24M1654397). *SIAM Journal on Scientific Computing*, 2025, 47(3): C655–C679.
10. Osnabrugge G, Leedumrongwatthanakun S, Vellekoop IM. [A convergent Born series for solving the inhomogeneous Helmholtz equation in arbitrarily large media](https://doi.org/10.1016/j.jcp.2016.06.034). *Journal of Computational Physics*, 2016, 322: 113–124.
11. Jia J, Liu X, Wang J, et al. [McMg: A Learned Phase-Space Multi-channel Multigrid Preconditioner for Helmholtz Equation](https://arxiv.org/abs/2606.30495). *arXiv*, 2026: 2606.30495.
12. Hao Z, Su C, Liu S, et al. [DPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training](https://proceedings.mlr.press/v235/hao24d.html). *International Conference on Machine Learning (ICML)*, 2024, PMLR 235: 17616–17635.
13. Liao W, Feng M, Zhang Z, et al. [G-RANS: Generalizable Residual-Aware Neural Solvers for Sparse Systems](https://proceedings.mlr.press/v306/liao26a.html). *International Conference on Machine Learning (ICML)*, 2026, PMLR 306: 73763–73776.
