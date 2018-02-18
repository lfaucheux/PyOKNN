import PyOKNN as ok

## Some print options related to numpy and pandas respectively
ok.npopts(8, True, 5e4)
ok.pdopts(5e4)

## Let's instantiate Presenter.
o = ok.Presenter(
    data_name = 'columbus',
    y_name    = 'CRIME',
    x_names   = ['INC', 'HOVAL'],
    id_name   = 'POLYID',
    verbose   = True,
    opverbose = True,
)

if 1:
    ## First (ols-like) charts and diagnosis.
    o.u_XACF_chart
    o.u_hull_chart


    ## ...
    o.u_XACF_chart_of(AR_ks=[1, 2, 4])
    o.u_hull_chart


    ## ...
    o.u_XACF_chart_of(MA_ks=[1, 2, 4])
    o.u_hull_chart


    ## ...
    o.u_XACF_chart_of(ER_ks=[1, 2, 4])
    o.u_hull_chart

    ## ...
    print o.summary()


if 0:
    ## Let's go for an MA{1,2,4}
    o.opverbose = False     # Printing minimizer's messages slows down processes.
    run_kwargs = {
        'plot_hist': True,  # Bootstrap distributions
        'plot_conv': True,  # Convergence plots
        'MA_ks'    : [1, 2, 4] ,
        'nbsamples': 10000, # Number of resampling iterations
    }
    o.PIs_computer(**run_kwargs)
