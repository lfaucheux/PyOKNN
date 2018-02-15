# [PyOKNN](https://github.com/lfaucheux/oknn) - A spatial lag operator proposal implemented in Python: *only the k-nearest neighbor* (oknn).

# Why

   [Fingleton (2009)](http://onlinelibrary.wiley.com/doi/10.1111/j.1538-4632.2009.00765.x/abstract) and [Corrado and Fingleton (2012)](http://onlinelibrary.wiley.com/doi/10.1111/j.1467-9787.2011.00726.x/abstract) remind us about the analogies between temporal and spatial processes, at least when considering their lag operators. In the spatial econometric (SE) case, the lag operator is always explicitly involved *via* the use of a <img src="https://latex.codecogs.com/gif.latex?n&space;\times&space;n" title="n \times n" /> matrix <img src="https://latex.codecogs.com/svg.latex?\boldsymbol{\mathrm{W}}" title="\boldsymbol{\mathrm{W}}" />, where <img src="https://latex.codecogs.com/gif.latex?n" title="n" /> is the number of interacting positions. The chosen space can be geographic, economic, social or of any other type. In the temporal case, which is seen as a space like no other given its inescapable anisotropic nature, the lag operator is in practice never explicitly considered. Any variable to lag, say, a <img src="https://latex.codecogs.com/gif.latex?n&space;\times&space;1" title="n \times 1" /> vector <img src="https://latex.codecogs.com/gif.latex?\boldsymbol{\mathrm{y}}" title="\boldsymbol{\mathrm{y}}" />, is formed over components that are prealably sorted according to their position on the timeline. This allows the lag-procedure to simply consist of offsetting down/up these components by a lag-determined number of rows, say, one row. In matrix terms, this offsetting procedure would be entirely equivalent to pre-multiplying an unsorted version of <img src="https://latex.codecogs.com/gif.latex?\boldsymbol{\mathrm{y}}" title="\boldsymbol{\mathrm{y}}" /> by a boolean <img src="https://latex.codecogs.com/gif.latex?n&space;\times&space;n" title="n \times n" /> matrix <img src="https://latex.codecogs.com/gif.latex?\boldsymbol{\mathrm{H}}" title="\boldsymbol{\mathrm{H}}" /> with <img src="https://latex.codecogs.com/gif.latex?1" title="1" />s indicating the immediate and unilateral proximity between temporal positions.

   The so-structured DGP thus involves <img src="https://latex.codecogs.com/gif.latex?\boldsymbol{\mathrm{H}}" title="\boldsymbol{\mathrm{H}}" /> as primarily observed, i.e. with no restructuring hypothesis or transformation. For each lag, this provides the statistician with a straightforward parameter space deﬁnition, whose knowledge of the exact boundary is important, both for estimation and inference [(Elhorst et al., 2012)](https://www.rug.nl/research/portal/publications/on-model-specification-and-parameter-space-definitions-in-higher-order-spatial-econometric-models(89205cfc-b2ad-4820-836c-2c4e919a9d11)/export.html).
   
   By opposition to the TS case, specifying <img src="https://latex.codecogs.com/svg.latex?\boldsymbol{\mathrm{W}}" title="\boldsymbol{\mathrm{W}}" /> involves a lot of arbitrariness. Apart from <img src="https://latex.codecogs.com/svg.latex?\boldsymbol{\mathrm{W}}" title="\boldsymbol{\mathrm{W}}" />’s non-nilpotency, these hypotheses deal with <img src="https://latex.codecogs.com/svg.latex?\boldsymbol{\mathrm{W}}" title="\boldsymbol{\mathrm{W}}" />’s isotropy [(Cressie, 1993)](http://doi.wiley.com/10.1002/9781119115151) and ﬁnding <img src="https://latex.codecogs.com/svg.latex?\boldsymbol{\mathrm{W}}" title="\boldsymbol{\mathrm{W}}" />’s true entrywise speciﬁcation through a very large number of competing ones, be it functional or binary. Some famous entrywise speciﬁcations are the negative exponential function [(Haggett, 1965)](http://www.abebooks.com/9780713151794/Locational-Analysis-Human-Geography-Peter-071315179X/plp?cm_sp=plped-_-1-_-image), the inverse-distance function [(Wilson, 1970)](http://www.worldcat.org/title/entropy-in-urban-and-regional-modelling/oclc/816930997?referer=null&ht=edition), the combined distance-boundary function [(Cliﬀ and Ord, 1973)](http://journals.sagepub.com/doi/abs/10.1177/030913259501900205) and the weighted logistic accessibility function [(Bodson and Peeters, 1975)](http://epn.sagepub.com/lookup/doi/10.1068/a070455). Binary weights speciﬁcations are either based on the k-nearest neighbor (knn), on the k-order of contiguity or on the radial distance. Then, to ensure the unique deﬁnition of any to-be-lagged variable in terms of the other variables of the model, <img src="https://latex.codecogs.com/svg.latex?\boldsymbol{\mathrm{W}}" title="\boldsymbol{\mathrm{W}}" /> is scaled depending on the choice one makes among three competing normalization techniques. The ﬁrst one makes <img src="https://latex.codecogs.com/svg.latex?\boldsymbol{\mathrm{W}}" title="\boldsymbol{\mathrm{W}}" /> row-stochastic, but does not necessarily preserve its symmetry. The second one pre- and post-multiplies <img src="https://latex.codecogs.com/svg.latex?\boldsymbol{\mathrm{W}}" title="\boldsymbol{\mathrm{W}}" /> by the negative square root of a diagonal matrix reporting its row-totals [(Cliﬀ and Ord, 1973)](http://journals.sagepub.com/doi/abs/10.1177/030913259501900205). The last one scales <img src="https://latex.codecogs.com/svg.latex?\boldsymbol{\mathrm{W}}" title="\boldsymbol{\mathrm{W}}" /> by its largest characteristic root [(Elhorst, 2001)](https://www.rug.nl/research/portal/publications/dynamic-models-in-space-and-time(3c6cb77b-b42c-4c64-b034-6f618d11a38f)/export.html). 
   
   But the choice of <img src="https://latex.codecogs.com/svg.latex?\boldsymbol{\mathrm{W}}" title="\boldsymbol{\mathrm{W}}" /> and of its transformation is not innocuous. For a maximum likelihood (ML) estimation to be consistent, the estimated spatial model must involve the true <img src="https://latex.codecogs.com/svg.latex?\boldsymbol{\mathrm{W}}" title="\boldsymbol{\mathrm{W}}" /> ([Dogan, 2013](http://www.mdpi.com/2225-1146/3/1/101); [Lee, 2004](http://repository.ust.hk/ir/Record/1783.1-32038)). When dealing with autoregressive disturbances, both estimators ML and spatial generalized moments (GM) ([Anselin, 2011](https://geodacenter.asu.edu/software/downloads/geodaspace); [Arraiz et al., 2010](http://onlinelibrary.wiley.com/doi/10.1111/j.1467-9787.2009.00618.x/abstract); [Drukker et al., 2013](http://dx.doi.org/10.1080/07474938.2013.741020); [Kelejian and Prucha, 2010](http://dx.doi.org/10.1016/j.jeconom.2009.10.025)) theoretically base their knowledge of unobservable innovations upon the knowledge of <img src="https://latex.codecogs.com/svg.latex?\boldsymbol{\mathrm{W}}" title="\boldsymbol{\mathrm{W}}" />. When facing endogeneity problems in non-autoregressive speciﬁcations and resorting to, *e.g.* [Kelejian and Prucha (1999)](http://onlinelibrary.wiley.com/doi/10.1111/1468-2354.00027/abstract)’s generalized moments estimator (GM), the deﬁnition of the exogeneity constrains heavily relies on <img src="https://latex.codecogs.com/svg.latex?\boldsymbol{\mathrm{W}}" title="\boldsymbol{\mathrm{W}}" />, which yields consistent and efﬁcient estimations for sure, but potentially not with respect to the true DGP. If resorting to the instrumental variables (IV) method – in which space is conceived as providing ideal instruments ([Das et al., 2003](http://dx.doi.org/10.1111/j.1435-5597.2003.tb00001.x); [Lee, 2003](http://www.tandfonline.com/doi/abs/10.1081/ETC-120025891); [Pinkse and Slade, 2010](http://onlinelibrary.wiley.com/doi/10.1111/j.1467-9787.2009.00645.x/abstract)) –, the strength of instruments is far from being ensured with <img src="https://latex.codecogs.com/svg.latex?\boldsymbol{\mathrm{W}}" title="\boldsymbol{\mathrm{W}}" /> in its most common speciﬁcation, i.e. whose lag consists of neighbors-averaging. Moreover, as discussed by [Gibbons and Overman (2012)](http://onlinelibrary.wiley.com/doi/10.1111/j.1467-9787.2012.00760.x/full), the inclusion of the product of higher powers of the spatial lag operator in the set of instruments is very likely to lead to a problem of colinearity, which in turn leads to the weaknesses of both identiﬁcation and instruments. Finally, when computing [LeSage and Pace (2009)](http://onlinelibrary.wiley.com/doi/10.1111/j.1751-5823.2009.00095_9.x/abstract)’s total direct and indirect eﬀects, the correctness of the true derivative of the regressand with respect to any spatially ﬁltered variable is a direct result of the correctness of <img src="https://latex.codecogs.com/svg.latex?\boldsymbol{\mathrm{W}}" title="\boldsymbol{\mathrm{W}}" />.
   
   Hence the proposal that follows, *i.e* a speciﬁcation method for the spatial lag operator whose properties are as close as possible to the ones of its time series (TS) counterpart, *i.e.* usable as primarily observed without modiﬁcations. Nonetheless we follow [Pinkse and Slade (2010, p.105)](http://onlinelibrary.wiley.com/doi/10.1111/j.1467-9787.2009.00645.x/abstract)’s recommendation of developing tools that are not simply extensions of familiar TS techniques to multiple dimensions. This is done so by proposing a speciﬁcation-method which is fully grounded on the observation of the empirical characteristics of space, while minimizing as much as possible the set of hypotheses that are required.
   
   Whereas the oknn speciﬁcation of <img src="https://latex.codecogs.com/gif.latex?\boldsymbol{\mathrm{W}}_k" title="\boldsymbol{\mathrm{W}}_k" /> is the strict spatial counterpart of the k-order TS lag operator, <img src="https://latex.codecogs.com/gif.latex?\boldsymbol{\mathrm{H}}^k" title="\boldsymbol{\mathrm{H}}^k" />, it had surprisingly never been proposed. The likely reason for this fact is the usual assumption of regular lattice, on which the autoregression structure superimposes. Frequently seen as an issue, the irregularity of the lattice is the rational for this speciﬁcation. Moreover, in realistic spatial conﬁgurations, the lattice regularity is the exception rather than the rule.
   
   This specification implies the transposition in space of the three-stage modeling approach of [Box and Jenkins (1976)](https://www.wiley.com/en-fr/Time+Series+Analysis:+Forecasting+and+Control,+5th+Edition-p-9781118675021) which consists of (i) identifying and selecting the model, (ii) estimating the parameters and (iii) checking the model. It follows that the models that are subject to selection in the present work are likely to involve a large number of parameters whose distributions, probably not symmetrical, are cumbersome to derive analytically. This is why in addition to the (normal-approximation-based) observed conﬁdence intervals, (non-adjusted and adjusted) bootstrap percentile intervals are implemented. However, the existence of ﬁxed spatial weight matrices prohibits the use of traditional bootstrapping methods. So as to compute (normal approximation or percentile-based) conﬁdence intervals for all the parameters, we use a special case of bootstrap method, namely [Lin et al. (2007)](https://www.researchgate.net/publication/228585605_Bootstrap_Test_Statistics_for_Spatial_Econometric_Models)’s hybrid version of residual-based recursive wild bootstrap. This method is particularly appropriate since it (i) "accounts for ﬁxed spatial structure and heteroscedasticity of unknown form in the data" and (ii) "can be used for model identiﬁcation (pre-test) and diagnostic checking (post-test) of a spatial econometric model". As mentioned above, non-adjusted percentile intervals as well as bias-corrected and accelerated (BCa) percentile intervals [(Efron and Tibshirani, 1993)](https://scholar.google.com/scholar_lookup?title=An%20introduction%20to%20the%20bootstrap&author=Efron&publication_year=1993) are implemented as well.
   

# How

## [Python2.7.+](https://www.python.org/ftp/python/2.7.14/python-2.7.14.msi) requirements

- **[matplotlib](https://matplotlib.org/)** *(tested under 1.4.3)*
- **[numdifftools](https://pypi.python.org/pypi/Numdifftools)**        *(tested under 0.9.20)* 
- **[numpy](http://www.numpy.org/)**        *(tested under 1.14.0)* 
- **[pandas](https://pandas.pydata.org/)**        *(tested under 0.22.0)* 
- **[scipy](https://www.scipy.org/)**        *(tested under 1.0.0)* 
- **[pysal](http://pysal.readthedocs.io/en/latest/)**        *(tested under 1.14.3)* 



## Installation

We are going to use a package management system used to install and manage software packages written in Python, namely [pip](https://en.wikipedia.org/wiki/Pip_(package_manager)). Open a session in your OS shell prompt and type

    pip install pyoknn

Or using a non python-builtin approach, namely [git](https://git-scm.com/downloads),

    git clone git://github.com/lfaucheux/PyOKNN.git
    cd PyOKNN
    python setup.py install

## Example usage:
The example that follows is done via the Python Shell. Let's first import the module `PyOKNN`.

    >>> import PyOKNN as ok  

We use [Anselin's Columbus OH 49 observation data set](https://nowosad.github.io/spData/reference/columbus.html). Since the data set is included in PyOKNN, there is no need to mention the path directory.

    >>> o = ok.Presenter(
    ...     data_name = 'columbus',
    ...     y_name    = 'CRIME',
    ...     x_names   = ['INC', 'HOVAL'],
    ...     id_name   = 'POLYID',
    ... )

Let's directly illustrate the main *raison d'être* of this package, i.e. which is about modelling the correlation structure of our OLS-like residuals. To do so, simply type

    >>> o.u_XACF_chart
    saved in  C:\data\Columbus.out\ER{0}AR{0}MA{0}[RESID][(P)ACF].png
    
`ER{0}AR{0}MA{0}[RESID][(P)ACF].png` looks like this

<img src="https://github.com/lfaucheux/PyOKNN/blob/master/PyOKNN/examples/ER%7B0%7DAR%7B0%7DMA%7B0%7D%5BRESID%5D%5B(P)ACF%5D.png?raw=true" width="425"/><img src="https://github.com/lfaucheux/PyOKNN/blob/master/PyOKNN/examples/ER%7B0%7DAR%7B0%7DMA%7B0%7D%5BRESID%5D%5BHULLS%5D.png?raw=true" width="425"/>

Be it in the ACF (upper dial) or in the PACF, we clearly have significant correlation at work through the lags 1, 2 and 4. Let's first think of it as global (thus considering the PACF) and go for an AR{1,2,4}.

    >>> o.u_XACF_chart_of(AR_ks=[1, 2, 4])
    Optimization terminated successfully.
             Current function value: 108.789436
             Iterations: 155
             Function evaluations: 292
    saved in  C:\data\Columbus.out\ER{0}AR{1,2,4}MA{0}[RESID][(P)ACF].png
<p align="center">
 <img src="https://github.com/lfaucheux/PyOKNN/blob/master/PyOKNN/examples/ER%7B0%7DAR%7B1,2,4%7DMA%7B0%7D%5BRESID%5D%5B(P)ACF%5D.png?raw=true" width="60%"/>
</p>

or thinking those as local, let's go for a MA{1,2,4}. 

    >>> o.u_XACF_chart_of(MA_ks=[1, 2, 4])
    Optimization terminated successfully.
             Current function value: 107.015463
             Iterations: 144
             Function evaluations: 268
    saved in  C:\data\Columbus.out\ER{0}AR{0}MA{1,2,4}[RESID][(P)ACF].png
<p align="center">
 <img src="https://github.com/lfaucheux/PyOKNN/blob/master/PyOKNN/examples/ER%7B0%7DAR%7B0%7DMA%7B1,2,4%7D%5BRESID%5D%5B(P)ACF%5D.png?raw=true" width="60%"/>
</p>

Thinking of CRIME variable as cointegrated through space with INC and HOVAL, let's go for a (partial) difference whose structure superimpose to the lags 1, 2 and 4.

    >>> o.u_XACF_chart_of(ER_ks=[1, 2, 4])
    Optimization terminated successfully.
             Current function value: 107.126738
             Iterations: 163
             Function evaluations: 304
    saved in  C:\Columbus.out\ER{1,2,4}AR{0}MA{0}[RESID][(P)ACF].png
<p align="center">
 <img src="https://github.com/lfaucheux/PyOKNN/blob/master/PyOKNN/examples/ER%7B1,2,4%7DAR%7B0%7DMA%7B0%7D%5BRESID%5D%5B(P)ACF%5D.png?raw=true" width="60%"/>
</p>


    
[ Forthcoming ] 
