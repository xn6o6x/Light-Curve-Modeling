---
title: Light Curve Modeling
tags: 天體物理討論班
---

**[Read on HackMD](https://hackmd.io/@juian/BkrwLpeCI)**

[TOC]

# References

## Code
- [ ] [PyTransit](https://github.com/hpparvi/PyTransit): Fast and easy-to-use tools for exoplanet transit light curve modelling with Python.
- [ ] [Exoplanet Light Curve Analysis](https://github.com/pearsonkyle/Exoplanet-Light-Curve-Analysis): A python package for modeling exoplanet light curves with nested sampling. The transit function uses the analytic expressions of Mandel and Agol 2002.
- [ ] [batman: Bad-Ass Transit Model cAlculatioN](https://www.cfa.harvard.edu/~lkreidberg/batman/index.html): A Python package for fast calculation of exoplanet transit light curves.
- [ ] [Lightkurve](http://docs.lightkurve.org/index.html): A friendly package for Kepler & TESS time series analysis in Python.
- [ ] [Harvard Transit Light Curve Tutorial](https://www.cfa.harvard.edu/~avanderb/tutorial/tutorial2.html)
- [ ] [Reduced Light Curves from K2 Campaigns 0 through 19 on MAST](https://www.cfa.harvard.edu/~avanderb/k2.html): Accessing and Downloading Kepler and K2 Data

## Literature
- [ ] [Basic light curve models](https://nexsci.caltech.edu/workshop/2012/talks/Agol_Sagan2012.pptx.pdf): PPT
- [ ] [Seager, S., & Mallen-Ornélas, G. 2002, ApJ, submitted (astro-ph/0206228)](https://arxiv.org/pdf/astro-ph/0206228.pdf): main analytical solution
- [ ] [Seager, S. & Mallen-Ornelas, G.. (2002). Extrasolar Planet Transit Light Curves and a Method to Select the Best Planet Candidates for Mass Follow-up. 34. ](https://arxiv.org/pdf/astro-ph/0210076.pdf): main analytical solution
- [ ] [Agol, E., Cowan, N., Bushong, J., Knutson, H., Charbonneau, D., Deming, D., & Steffen, J. (2008). Transits and secondary eclipses of HD 189733 with Spitzer. Proceedings of the International Astronomical Union, 4(S253), 209-215. doi:10.1017/S1743921308026422](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/E3B29F85EC33CB8EE27305C0659CA06B/S1743921308026422a.pdf/transits_and_secondary_eclipses_of_hd_189733_with_spitzer.pdf): Mid-infrared transit almost no limb darkening
- [ ] [胡瑞華（2004）。在疏散星團中尋找系外行星與變星。國立中央大學天文研究所碩士論文。](https://hdl.handle.net/11296/j2qtxj)

# Assumptions

1. The planet orbit is circular (valid for tidally-circularized extrasolar planets);
> 对于太阳系外行星合理，因为短周期行星预期有圆形轨道。
2. $M_p\ll M_∗$ and the companion is dark compared to the central star;
3. The stellar mass-radius relation is known;
> 恒星的流体静力学平衡 + 状态方程（多方形式）→ Lane-Emden 方程，数值求解得到质量-半径关系。
> 
> 唯一有可能出错的情况是混合星（blended star）的存在，例如，来自恒星的物理伴星。
4. The light comes from a single star, rather than from two or more blended stars.
5. The eclipses have flat bottoms. This implies that the companion is fully superimposed on the central star’s disk and requires that the data is in a band pass where limb darkening is negligible;
6. The period can be derived from the light curve (e.g., the two observed eclipses are consecutive).

# 5 Physical Parameters

$M_∗$, $R_∗$, $a$, $i$, and $R_p$

# 4 Observable Parameters

$\Delta F$ , $t_T$ , $t_F$ , and $P$

<a href="https://arxiv.org/pdf/astro-ph/0206228.pdf"><img src="https://i.imgur.com/dX0ZBoN.png" style="width: 500px;"/></a>

1. Full duration $t_F$ (second to third contact)
2. Total duration $t_T$ (first to fourth contact)

# 4 Combinations of Parameters

$\dfrac{R_p}{R_∗}$, $b=\dfrac{a\cos i}{R_*}$, $\dfrac{a}{R_*}$, and $\dfrac{\rho_*}{\rho_\odot}$

## Impact Parameter

$b=\dfrac{a \cos i}{R_{*}}$，$0\le b\le 1$
<a href="https://www.paulanthonywilson.com/exoplanets/exoplanet-detection-techniques/the-exoplanet-transit-method/"><img src="https://i.imgur.com/z6QuwaK.png" style="width: 250px;"/></a>

# 3 Geometrical Equations

$\Delta F=\left(\dfrac{R_{p}}{R_{*}}\right)^2\quad...\ (1)$

$t_T=P \dfrac{\alpha}{2 \pi}=\dfrac{P}{\pi} \arcsin\left(\dfrac{\sqrt{\left(1+R_{p}/R_{*}\right)^{2}-b^{2}}}{a\sin i/R_*}\right)\quad...\ (2)$

$\dfrac{t_F}{t_T}=\dfrac{\arcsin\left(\dfrac{\sqrt{\left(1-R_{p}/R_{*}\right)^{2}-b^{2}}}{a\sin i/R_*}\right)}{\arcsin\left(\dfrac{\sqrt{\left(1+R_{p}/R_{*}\right)^{2}-b^{2}}}{a\sin i/R_*}\right)}\quad...\ (3)$

<a href="https://www.paulanthonywilson.com/exoplanets/exoplanet-detection-techniques/the-exoplanet-transit-method/"><img src="https://i.imgur.com/FIfQ3th.jpg" style="width: 800px;"/></a>

$2 l=2 \sqrt{\left(R_{*}+R_{p}\right)^{2}-\left(b R_{*}\right)^{2}}$，$\sin \left(\dfrac{\alpha}{2}\right)=\dfrac{l}{a\sin i}$

# Kepler’s Third Law

$P^{2}=\dfrac{4 \pi^{2} a^{3}}{G\left(M_{*}+M_{p}\right)}\sim\dfrac{4 \pi^{2} a^{3}}{GM_{*}}$

# Lane–Emden equation

- 恒星的流体静力学平衡：$\dfrac{dP}{dr} = -\dfrac{Gm\rho}{r^2}$

- 状态方程（多方形式）：$P=P(\rho)=K\rho^\gamma$，$\gamma=1+1/n$

- Lane-Emden 方程：$\dfrac{1}{\xi^2} \dfrac{d}{d\xi} \left({\xi^2 \dfrac{d\theta}{d\xi}}\right) + \theta^n = 0$


> $\rho=\rho_{0} \theta^{n}\quad r=a \xi \quad a=\sqrt{\dfrac{(n+1) K}{4 \pi G} \rho_{0}^{1/n-1}}\quad {\displaystyle \theta (\xi )=\sum _{n=0}^{\infty }a_{n}\xi ^{n}}$
> 
> $\theta(0)=1$, $\theta'(0)=0$
- 数值求解得到

$\begin{array}{l}R=a \xi_{1}=\left[\dfrac{(n+1) K}{4 \pi G}\right]^{1 / 2} \rho_{0}^{(1-n) / 2 n} \xi_{1} \\ \displaystyle M=\int_{0}^{R} 4 \pi r^{2} \rho \mathrm{d} r=4 \pi\left[\frac{(n+1) K}{4 \pi G}\right]^{3 / 2} \rho_{0}^{(3-n) / 2 n} \xi_{1}^{2}\left|\theta^{\prime}\left(\xi_{1}\right)\right|\end{array}$

消去 $\rho_0$

$M=4\pi R^{(3-n)/(1-n)}\left[\dfrac{(n+1) K}{4 \pi G}\right]^{n /(n-1)} \xi_{1}^{(3-n) /(n-1)} \xi_{1}^{2}\left|\theta^{\prime}\left(\xi_{1}\right)\right|$

$R=kM^x$

> Here $k$ is a constant coefficient for each stellar sequence (main sequence, giants, etc.) and $x$ describes the power law of the sequence (e.g., $k=1$ and $x≃0.8$ for F–K main sequence stars (Cox 2000)).
 
# Solution

联立 $\tau_T=\dfrac{\sqrt{(1+\sqrt{\Delta F})^2-b^2}}{a\sin i}$ 和 $\tau_F=\dfrac{\sqrt{(1-\sqrt{\Delta F})^2-b^2}}{a\sin i}$ 得到

$\displaystyle b=\dfrac{a}{R_*}\cos i=\sqrt{\dfrac{(1-\sqrt{\Delta F})^{2}-\left(\dfrac{\tau_F}{\tau_T}\right)^2(1+\sqrt{\Delta F})^{2}}{1-\left(\dfrac{\tau_F}{\tau_T}\right)^2}}$

> $\tau_F=\sin\dfrac{t_{F} \pi}{P}$，$\tau_T=\sin\dfrac{t_{T} \pi}{P}$
 
将 $a\sin i=\sqrt{a^2-b^2}$ 代入 $(2)$

$\dfrac{a}{R_{*}}=\dfrac{\sqrt{(1+\sqrt{\Delta F})^{2}-b^{2}\left(1-\tau_T^2\right)}}{\tau_T}$

又由 $(4)$ 得到

$a=\left[\dfrac{P^{2} G M_{*}}{4 \pi^{2}}\right]^{1/3}$

有

$\displaystyle \frac{\rho_*}{\rho_\odot}=\frac{M_*/M_\odot}{(R_*/R_\odot)^3}=\left[\frac{4\pi^2}{P^2G}\right]\left[\dfrac{(1+\sqrt{\Delta F})^{2}-b^{2}\left(1-\tau_T^2\right)}{\tau_T^2}\right]^{3/2}$


由 $(5)$ 得到

1. $\displaystyle \frac{M_{*}}{M_{\odot}}=\left[k^{3} \frac{\rho_{*}}{\rho_{\odot}}\right]^{\frac{1}{1-3 x}}$

2. $\displaystyle \frac{R_{*}}{R_{\odot}}=\left[k^{1/x} \frac{\rho_{*}}{\rho_{\odot}}\right]^{\frac{x}{1-3 x}}$

3. $a=\left[\dfrac{P^{2} G M_{*}}{4 \pi^{2}}\right]^{1/3}$

4. $i=\arccos\left(b\dfrac{R_{*}}{a}\right)$

5. $\dfrac{R_{p}}{R_{\odot}}=\dfrac{R_{*}}{R_{\odot}} \sqrt{\Delta F}$

# Simplified Equations

If $R_*\ll a$ (is equivalent to $\cos i\ll 1$), $\tau_T\approx\dfrac{t_T}{P}\pi$, $\tau_F\approx\dfrac{t_F}{P}\pi$, $\sin i\approx1$,

直接得到

$\displaystyle b=\sqrt{\dfrac{(1-\sqrt{\Delta F})^{2}-\left(\dfrac{t_F}{t_T}\right)^2(1+\sqrt{\Delta F})^{2}}{1-\left(\dfrac{t_F}{t_T}\right)^2}}$

:::info
$\Delta F$ 越大（越深），$b$ 越小
:::

$(2)$ 变成 $t_T=\dfrac{P}{\pi} \left(\dfrac{\sqrt{\left(1+\sqrt{\Delta F}\right)^{2}-b^{2}}}{a/R_*}\right)$

同理 $t_F=\dfrac{P}{\pi} \left(\dfrac{\sqrt{\left(1-\sqrt{\Delta F}\right)^{2}-b^{2}}}{a/R_*}\right)$

两式平方相减

$\dfrac{a}{R_{*}}=\dfrac{2 P}{\pi} \dfrac{\Delta F^{1 / 4}}{\left(t_{T}^{2}-t_{F}^{2}\right)^{1 / 2}}$

进而有

$\dfrac{\rho_{*}}{\rho_{\odot}}=\dfrac{32}{G \pi} P \dfrac{\Delta F^{3 / 4}}{\left(t_{T}^{2}-t_{F}^{2}\right)^{3 / 2}}$

# [Limb Darkening](https://en.wikipedia.org/wiki/Limb_darkening)

![](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Limb_darkening_geometry.svg/171px-Limb_darkening_geometry.svg.png)

${\displaystyle \cos \psi ={\frac {\sqrt {\cos ^{2}\theta -\cos ^{2}\Omega }}{\sin \Omega }}={\sqrt {1-\left({\frac {\sin \theta }{\sin \Omega }}\right)^{2}}}=\mu}$

${\displaystyle {\frac {I(\psi )}{I(0)}}=\sum _{k=0}^{N}a_{k}\,{\textrm {cos}}^{k}(\psi )}$

$I(0)$ is the central intensity, so we must have ${\displaystyle \sum _{k=0}^{N}a_{k}=1}$

For example, for a Lambertian radiator (no limb darkening) we will have all $a_k=0$ except $a_0=1$.

As another example, for the Sun at 550 nm, the limb darkening is well expressed by

$N=2$
${\displaystyle a_{0}=1-a_{1}-a_{2}=0.3}$
${\displaystyle a_{1}=0.93}$
${\displaystyle a_{2}=-0.23}$

The equation for limb darkening is sometimes more conveniently written as ${\displaystyle {\frac {I(\psi )}{I(0)}}=1+\sum _{k=1}^{N}A_{k}\,(1-\cos(\psi ))^{k}}$

$\mu=\cos\psi$

$x=R\sin(\psi-\theta)$

- Linear: $\dfrac{I(\mu)}{I(1)}=1-u(1-\mu)$

- Quadratic: $\dfrac{I(\mu)}{I(1)}=1-a(1-\mu)-b(1-\mu)^{2}$

- 3 Parameters non-linear: $\dfrac{I(\mu)}{I(1)}=1-c_{2}(1-\mu)-c_{3}\left(1-\mu^{3 / 2}\right)-c_{4}\left(1-\mu^{2}\right)$

- 4 Parameters non-linear: $\dfrac{I(\mu)}{I(1)}=1-c_{1}\left(1-\mu^{1 / 2}\right)-c_{2}(1-\mu)-c_{3}\left(1-\mu^{3 / 2}\right)-c_{4}\left(1-\mu^{2}\right)$

> [Kepler Stellar Limb-Darkening Coefficients](https://pages.jh.edu/~dsing3/David_Sing/Limb_Darkening.html)

The mean intensity $I_m$ is the integral of the intensity over the disk of the star divided by the solid angle subtended by the disk

${\displaystyle I_{m}={\frac {\int I(\psi )\,d\omega }{\int d\omega }}}$，$d\omega = \sin \theta d\theta d\psi$，$0≤\psi≤2\pi$ and $0≤\theta≤\Omega$

$\displaystyle I_{m}={\frac {\int _{\cos \Omega }^{1}I(\psi )\,d\cos \theta }{\int _{\cos \Omega }^{1}d\cos \theta }}={\frac {\int _{\cos \Omega }^{1}I(\psi )\,d\cos \theta }{1-\cos \Omega }}$

For an observer at infinite distance from the star, ${\displaystyle d\cos \theta }$ can be replaced by ${\displaystyle \sin ^{2}\Omega \cos \psi \,d\cos \psi }$, so we have

$\displaystyle I_{m}={\frac {\int _{0}^{1}I(\psi )\cos \psi \,d\cos \psi }{\int _{0}^{1}\cos \psi \,d\cos \psi }}=2\int _{0}^{1}I(\psi )\cos \psi \,d\cos \psi$

$\displaystyle {\frac {I_{m}}{I(0)}}=2\sum _{k=0}^{N}{\frac {a_{k}}{k+2}}$

![](https://i.imgur.com/ijmO8le.png)

![](https://i.imgur.com/9RrvWIY.png)

<a href="http://exoplanet-diagrams.blogspot.com/2015/07/transits-limb-darkening-hd-209458-b-in.html"><img src="http://2.bp.blogspot.com/-2bwFgSzo2_c/VbktcgNZe4I/AAAAAAAAC7A/uTFtWpBlKgk/s1600/hd209458-plot.png"
style="width: 300px;"/></a>

:::success
短波段的临边昏暗效应比较明显，且短波段的曲线更深（$b$ 更小）。
:::

# To Do

- Errors
- Blended Stars

# Bayesian parameter estimation

It is common to adopt a Bayesian attitude, in which the parameters are viewed as random variables whose probability distributions (“posteriors”) are constrained by the data. This can be done in a convenient and elegant fashion us- ing the Monte Carlo Markov Chain (MCMC) method, in which a chain of points is created in parameter space using a few simple rules that ensure the collection of points will converge toward the desired posterior. This method gives the full multidimensional joint probability distribution for all the parameters, rather than merely giving individual er- ror bars, making it easy to visualize any fitting degeneracies and to compute posteriors for any combination of parame- ters that may be of interest.

## [似然函数](https://zh.wikipedia.org/zh-cn/似然函数)（Likelihood function）

- 概率（或然性）：用于在已知一些参数的情况下，预测接下来在观测上所得到的结果.
- 似然性：用于在已知某些观测所得到的结果时，对有关事物之性质的参数进行估值。

似然函数可以理解为条件概率的逆反。在已知某个参数 B 时，事件 A 会发生的概率写作：

$P(A\mid B)={\dfrac{P(A\cap B)}{P(B)}}$

利用贝叶斯定理，

$P(B\mid A)={\dfrac{P(A\mid B)\;P(B)}{P(A)}}$

因此，我们可以反过来构造表示似然性的方法：已知有事件 A 发生，运用似然函数 $L(B\mid A)$ 估计参数 B 的可能性。

> 似然函数并不是几率密度函数，且不要求满足归一性 $\displaystyle\sum _{{b\in {\mathcal {B}}}}P(A\mid B=b)=1$

:::success
对于一个概率模型，如果其参数为 $\theta$，那么在给定观察数据 $x$ 时，该参数的似然方程被定义为：$L(\theta\mid x)=P(x\mid\theta)$
:::

考虑投掷一枚硬币的实验。通常来说，已知掷出一枚“公平的硬币”，即正面（Head）朝上的概率为 $p_{H}=0.5$，便可以知道投掷若干次后出现各种结果的可能性。

在统计学中，我们关心的是在已知一系列投掷的结果时，关于硬币投掷时正面朝上的可能性的信息。我们可以建立一个统计模型：假设硬币投出时会有 $p_{H}$ 的概率正面朝上，而有 $1-p_{H}$ 的概率反面朝上。

这时，通过观察**已发生**的两次投掷，条件概率可以改写成似然函数：

$L(p_{H}\mid {\mbox{HH}})=P({\mbox{HH}}\mid p_{H})$

对于取定的似然函数，在观测到两次投掷都是正面朝上时，$p_{H}=0.5$ 的似然性是 0.25。

${\displaystyle L(p_{H}\mid {\mbox{HH}})=P({\mbox{HH}}\mid p_{H}=0.5)=0.25}$

> 反之并不成立，即当似然函数为 0.25 时不能推论出  $p_{H}=0.5$。

如果考虑 $p_{H}=0.6$，那么似然函数的值也会改变

${\displaystyle L(p_{H}\mid {\mbox{HH}})=P({\mbox{HH}}\mid p_{H}=0.6)=0.36}$

这说明，如果参数 $p_{H}$ 的取值变成 0.6 的话，结果观测到连续两次正面朝上的概率要比假设 $p_{H}=0.5$ 时更大。也就是说，参数 $p_{H}$ 取成 0.6 要比取成 0.5 更有说服力，更为“合理”。

:::success
对同一个似然函数，其所代表的模型中，某项参数值具有多种可能，但如果存在一个参数值，使得它的函数值达到最大的话，那么这个值就是该项参数最为“合理”的参数值。
:::

如果观测到的是三次投掷硬币，头两次正面朝上，第三次反面朝上，那么似然函数将会是：

${\displaystyle L(\theta \mid {\mbox{HHT}})=P({\mbox{HHT}}\mid p_{H}=\theta )=\theta ^{2}(1-\theta )}$

这时候，似然函数的最大值将会在 $p_{H}={\dfrac{2}{3}}$ 的时候取到。

![](https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/LikelihoodFunctionAfterHHT.png/600px-LikelihoodFunctionAfterHHT.png)

## [先验概率](https://zh.wikipedia.org/zh-cn/先验概率)（Prior probability）

表达某一不确定量的不确定性概率分布。它旨在描述这个不确定量的不确定程度，而不是这个不确定量的随机性。

## [后验概率](https://zh.wikipedia.org/zh-cn/后验概率)（Posterior probability）

给定证据 $X$ 后，参数 $\theta$ 的概率 ${\displaystyle p(\theta|X)}$

后验概率 $p(\theta|X)$ $\propto$ 可能性 $p(X|\theta)$ $\times$ 先验概率 $p(\theta)$

归一化：${\displaystyle p(\theta |x)={\frac {p(x|\theta )p(\theta )}{p(x)}}}$

## Implementation

This first tutorial covers the simple case of an exoplanet system characterisation based on a single photometric timeseries of an exoplanet transit (transit light curve). The system characterisation is a *parameter estimation* problem, where we assume we have an adequate model to describe the observations, and we want to infer the model parameters with their uncertainties.

> 基于单个光变曲线的时间序列表征（characterisation）系外行星系统的是一个参数估计（parameter estimation）问题。我们假设我们有一个足够的模型来描述观测值，并且我们要推断模型参数的不确定性。

We take a [*Bayesian*](http://en.wikipedia.org/wiki/Bayesian_probability) approach to the parameter estimation, where we want to estimate the [*posterior probability*](http://en.wikipedia.org/wiki/Posterior_probability) for the model parameters given their [*prior probabilities*](http://en.wikipedia.org/wiki/Prior_probability) and a set of observations. The posterior probability density given a parameter vector $\theta$ and observational data $D$ is described by the [*Bayes' theorem*](http://en.wikipedia.org/wiki/Bayes%27_theorem) as

$$
P(\theta|D) = \frac{P(\theta) P(D|\theta)}{P(D)}, \qquad P(D|\theta) = \prod P(D_i|\theta),
$$

where $P(\theta)$ is the prior, $P(D|\theta)$ is the [*likelihood*](http://en.wikipedia.org/wiki/Likelihood_function) for the data, and $P(D)$ is a [*normalising factor*](http://en.wikipedia.org/wiki/Marginal_likelihood) we don't need to bother with during [MCMC](https://zh.wikipedia.org/zh-cn/马尔可夫链蒙特卡洛)-based parameter estimation.

> 根据模型参数的先验概率 $P(\theta)$ 和一组观测值 $D$ 来估计模型参数的后验概率 $P(\theta|D)$。

The likelihood is a product of individual observation probabilities, and has the unfortunate tendency to end up being either very small or very big. This causes computational headaches, and it is better to work with log probabilities instead, so that

$$
\log P(\theta|D) = \log P(\theta) + \log P(D|\theta),  \qquad \log P(D|\theta) = \sum \log P(D_i|\theta)
$$

where we have omitted the $P(D)$ term from the posterior density.

> 取对数，忽略用来归一化的 $P(D)$ 项。

Now we still need to decide our likelihood density. If we can assume normally distributed white noise--that is, the errors in the observations are independent and identically distributed--we end up with a log likelihood function

$$
 \log P(D|\theta) = -N\log(\sigma) -\frac{N\log 2\pi}{2} - \sum_{i=0}^N \frac{(o_i-m_i)^2}{2\sigma^2},
$$

where $N$ is the number of datapoints, $\sigma$ is the white noise standard deviation, $o$ is the observed data, and $m$ is the model. [proof](https://www.statlect.com/fundamentals-of-statistics/normal-distribution-maximum-likelihood)

> 假定白噪声（[独立同分布 I.I.D.](https://zh.wikipedia.org/zh-cn/独立同分布)）$\{X_n\}$
> 
> The probability density function $f_{X}\left(x_{j}\right)=\left(2 \pi \sigma_{0}^{2}\right)^{-1 / 2} \exp \left(-\dfrac{1}{2} \dfrac{\left(x_{j}-\mu_{0}\right)^{2}}{\sigma_{0}^{2}}\right)$
> 
> The likelihood function $\displaystyle \begin{aligned} L\left(\mu, \sigma^{2} ; x_{1}, \ldots, x_{n}\right) &=\prod_{j-1}^{n} f_{X}\left(x_{j} ; \mu, \sigma^{2}\right) \\ &=\prod_{j=1}^{n}\left(2 \pi \sigma^{2}\right)^{-1 / 2} \exp \left(-\frac{1}{2} \frac{\left(x_{j}-\mu\right)^{2}}{\sigma^{2}}\right) \\ &=\left(2 \pi \sigma^{2}\right)^{-n / 2} \exp \left(-\frac{1}{2 \sigma^{2}} \sum_{j=1}^{n}\left(x_{j}-\mu\right)^{2}\right) \end{aligned}$
> 
> The log-likelihood function $\displaystyle l\left(\mu, \sigma^{2} ; x_{1}, \ldots, x_{n}\right)=-\frac{n}{2} \ln (2 \pi)-\frac{n}{2} \ln \left(\sigma^{2}\right)-\frac{1}{2 \sigma^{2}} \sum_{j=1}^{n}\left(x_{j}-\mu\right)^{2}$

# BLS

- [x] [Box least squares (BLS) periodogram — Astropy v4.0.1](https://docs.astropy.org/en/stable/timeseries/bls.html)

BLS(Box-fitting Least Squares) is a box-fitting algorithm that analyzes stellar photometric time series to search for periodic transits of extrasolar planets. It searches for signals characterized by a periodic alternation between two discrete levels, with much less time spent at the lower level.

The BLS method finds transit candidates by modeling a transit as a periodic upside down top hat with four parameters: `period`, `duration`, `depth`, and a `reference time`. In this implementation, the reference time is chosen to be the mid-transit time of the first transit in the observational baseline. These parameters are shown in the following sketch:

<a href="https://docs.astropy.org/en/stable/timeseries/bls.html"><img src="https://docs.astropy.org/en/stable/timeseries/bls-1.svg" style="width: 500px;"/></a>

Assuming that the uncertainties on the measured flux are known, independent, and Gaussian, the maximum likelihood in-transit flux can be computed as $\displaystyle y_{\mathrm{in}}=\frac{\sum_{\mathrm{in}} y_{n} / \sigma_{n}^{2}}{\sum_{\mathrm{in}} 1 / \sigma_{n}^{2}}$ where $y_n$ are the brightness measurements, $\sigma_n$ are the associated uncertainties, and both sums are computed over the in-transit data points.

Similarly, the maximum likelihood out-of-transit flux is $\displaystyle y_{\mathrm{out}}=\frac{\sum_{\mathrm{out}} y_{n} / \sigma_{n}^{2}}{\sum_{\mathrm{out}} 1 / \sigma_{n}^{2}}$

The log likelihood of a transit model (maximized over depth) at a given `period` $P$, `duration` $\tau$, and `reference time` $t_0$ is

$\displaystyle \log \mathcal{L}\left(P, \tau, t_{0}\right)=-\frac{1}{2} \sum_{\mathrm{in}} \frac{\left(y_{n}-y_{\mathrm{in}}\right)^{2}}{\sigma_{n}^{2}}-\frac{1}{2} \sum_{\mathrm{out}} \frac{\left(y_{n}-y_{\mathrm{out}}\right)^{2}}{\sigma_{n}^{2}}+\mathrm{constant}$

This equation might be familiar because it is proportional to the “chi squared” $\chi^2$ for this model and this is a direct consequence of our assumption of Gaussian uncertainties.

Box-Fitting 基本的做法为用最小平方法拟合不同的参数组成的 step function，找出最佳的参数组合，共有五个参数：周期 $P_0$、掩星持续时间与周期的比例 $q$、掩星发生时的光度 $L$、主星原本的光度 $H$、掩星发生的时间 $t_0$。

假设所以资料点为 $\{\tilde x_i\}$，相对应的权重为 $\{\tilde w_i\equiv\dfrac{\sigma_i^{-2}}{\sum_{j=1}^{n}\sigma_j^{-2}}\},i=1,2,3,...,n,$

$\displaystyle\mathcal{D}=\sum_\text{in} \tilde{w}_{i}\left(\tilde{x}_{i}-\hat{L}\right)^{2}+\sum_\text{out} \tilde{w}_{i}\left(\tilde{x}_{i}-\hat{H}\right)^{2}$

取 $\mathcal{D}$ 最小，即可得到最佳参数。

## Boxcar Transit Model

<a href="https://nexsci.caltech.edu/workshop/2012/talks/Agol_Sagan2012.pptx.pdf"><img src="https://i.imgur.com/wNJUMGR.png" style="width: 500px;"/></a>

Box-car/pulse/top-hat transit shape is useful for transit searches, e.g. BLS(Box-fitting Least Squares) or QATS(The Quasiperiodic Automated Transit Search Algorithm).

# [Kepler](https://keplerscience.arc.nasa.gov/objectives.html)

The Kepler spacecraft launched in March 2009 and spent a little over four years monitoring more than 150,000 stars in the Cygnus-Lyra region with continuous 30-min or 1-min sampling. The primary science objective of the Kepler mission was transit-driven exoplanet detection with an emphasis on terrestrial (R < 2.5 R Earth) planets located within the habitable zones of Sun-like stars.

## K2
Extending Kepler's power to the ecliptic

The loss of a second of the four reaction wheels on board the Kepler spacecraft in May 2013 brought an end to Kepler's four plus year science mission to continuously monitor more than 150,000 stars to search for transiting exoplanets. Developed over the months following this failure, the K2 mission represents a new concept for spacecraft operations that enables continued scientific observations with the Kepler space telescope. K2 became fully operational in June 2014 and is expected to continue operating until 2017 or 2018.

## [K2 observing](https://keplerscience.arc.nasa.gov/k2-observing.html)

The broad photometric bandpass has a half-maximum transmission range of 430 to 840 nm. The instrument has neither changeable filters, dispersing elements, nor a shutter.

K2 observations entail a series of sequential observing ["Campaigns"](https://keplerscience.arc.nasa.gov/k2-fields.html) of fields distributed around the ecliptic plane. Each ecliptic Campaign is limited by Sun angle constraints to a duration of approximately 80 days as illustrated in the image below (Howell et al. 2014). Therefore, four to five K2 Campaigns can be performed during each 372-day orbit of the spacecraft.

![](https://keplerscience.arc.nasa.gov/images/k2/footprint-all-campaigns.png?nocache=1)

![](https://keplerscience.arc.nasa.gov/images/k2_explained_25nov_story.jpg)

## [Kepler and K2 data processing pipeline](https://keplerscience.arc.nasa.gov/pipeline.html)

The original mission observed stars for about four years searching for planets, so Kepler data often is a nearly continuous, high quality light curve for four years. K2, on the other hand, looks at stars for only about 80 days at a time. K2 data also has more data quirks and glitches than data from the original Kepler mission.

> For Kepler data only, the pipeline also included the following elements: **Transiting Planet Search (TPS)** and **Data Validation (DV)**

### Calibration (CAL)
- Bias level
- Dark current
- Smear
- Gain
- Undershoot
- Flat field

### Photometric Analysis (PA)
- Barycentric time correction
- "Argabrightening" event detection
- Cosmic ray cleaning
- Background removal
- Aperture photometry
- Source centroids
- Astrometric solution
- Computation of metrics

In the currently exported FITS light curve files, the output of PA is labeled "raw" flux to distinguish it from light curves which have been corrected for systematic effects in the subsequent PDC software module.

### Pre-search Data Conditioning (PDC)
- Data anomaly flagging
- Resample ancillary spacecraft data
- Identification and correction of discontinuities
- Identify variable stars
- Identify astrophysical events: e.g., giant planet transits, stellar eclipses, flares and microlensing events
- Systematic error correction for quiet stars: remove correlated trends
- Systematic error correction for variable stars
- Correct excess flux
- Identification of outlying data points

![](https://keplerscience.arc.nasa.gov/images/PDC_example3_quietstar_drn5.jpg)