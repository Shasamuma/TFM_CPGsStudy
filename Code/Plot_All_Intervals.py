def Plot_All_Intervals(t, V_PD, V_LP, slices_PD, slices_LP, x_min=0, x_max = 50000):
    import matplotlib.pyplot as plt
    import numpy as np

    factor_d2 = 1.5
    factor_d3 = 4.2

    t_cut = t[x_min:x_max]
    ## Vertical lines IBI LP
    plt.annotate("",
                xy=(t_cut[slices_LP[0][-1]], -2.05), xycoords='data',
                xytext=(t_cut[slices_LP[0][-1]], 2.55), textcoords='data',
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3", linestyle='dotted',color='turquoise'))
    plt.annotate("",
                xy=(t_cut[slices_LP[1][0]], -2.55), xycoords='data',
                xytext=(t_cut[slices_LP[1][0]], 2.55), textcoords='data',
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3", linestyle='dotted',color='turquoise'))


    ## Vertical lines IBI PD
    plt.annotate("",
                xy=(t_cut[slices_PD[0][-1]], V_PD[slices_PD[0][-1]]+factor_d3), xycoords='data',
                xytext=(t_cut[slices_PD[0][-1]], 2.05), textcoords='data',
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3", linestyle='dotted',color='lightsalmon'))
    plt.annotate("",
                xy=(t_cut[slices_PD[1][0]], -2.55), xycoords='data',
                xytext=(t_cut[slices_PD[1][0]], 2.05), textcoords='data',
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3", linestyle='dotted',color='lightsalmon'))


    ## Vertical line LP period
    plt.annotate("",
                xy=(t_cut[slices_LP[0][0]], -2.55), xycoords='data',
                xytext=(t_cut[slices_LP[0][0]], 1.55), textcoords='data',
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3", linestyle='dotted',color='turquoise'))


    ## Vertical line PD period
    plt.annotate("",
                xy=(t_cut[slices_PD[0][0]], V_PD[slices_PD[0][0]]+factor_d3), xycoords='data',
                xytext=(t_cut[slices_PD[0][0]], 1.05), textcoords='data',
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3", linestyle='dotted',color='lightsalmon'))


    ## Horizontal line and tag LP period
    plt.annotate("",
                xy=(t_cut[slices_LP[1][0]], 1.5), xycoords='data',
                xytext=(t_cut[slices_LP[0][0]], 1.5), textcoords='data',
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))
    plt.annotate("LP Period",
                xy=(t_cut[slices_LP[1][0]], -0.8), xycoords='data',
                xytext=((t_cut[slices_LP[0][0]]+t_cut[slices_LP[1][0]])*2/5, 1.6), textcoords='data')


    ## Vertical line burst PD
    plt.annotate("",
                xy=(t_cut[slices_PD[1][-1]], V_PD[slices_PD[1][-1]]+factor_d3), xycoords='data',
                xytext=(t_cut[slices_PD[1][-1]], -2.05), textcoords='data',
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3", linestyle='dotted',color='lightsalmon'))


    ## Horizontal line and tag PD period
    plt.annotate("",
                xy=(t_cut[slices_PD[0][0]], 1), xycoords='data',
                xytext=(t_cut[slices_PD[1][0]], 1), textcoords='data',
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))
    plt.annotate("PD Period",
                xy=(t_cut[slices_PD[0][0]], 1), xycoords='data',
                xytext=((t_cut[slices_PD[1][0]]+t_cut[slices_PD[0][0]])*2./5, 1.1), textcoords='data')


    ## Horizontal line and tag IBI LP
    plt.annotate("",
                xy=(t_cut[slices_LP[0][-1]], 2.5), xycoords='data',
                xytext=(t_cut[slices_LP[1][0]], 2.5), textcoords='data',
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))
    plt.annotate("IBI LP",
                xy=(t_cut[slices_LP[0][-1]], 2.5), xycoords='data',
                xytext=((t_cut[slices_LP[0][-1]]+t_cut[slices_LP[1][0]])*2.2/5, 2.6), textcoords='data')


    ## Horizontal line and tag IBI PD
    plt.annotate("",
                xy=(t_cut[slices_PD[0][-1]], 2), xycoords='data',
                xytext=(t_cut[slices_PD[1][0]], 2), textcoords='data',
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))
    plt.annotate("IBI PD",
                xy=(t_cut[slices_PD[0][-1]], 2), xycoords='data',
                xytext=((t_cut[slices_PD[0][-1]]+t_cut[slices_PD[1][0]])*2.1/5, 2.1), textcoords='data')


    ## Horizontal line and tag Burst LP
    plt.annotate("",
                xy=(t_cut[slices_LP[0][0]], -1.5), xycoords='data',
                xytext=(t_cut[slices_LP[0][-1]], -1.5), textcoords='data',
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))
    plt.annotate("Burst LP",
                xy=(t_cut[slices_LP[0][0]], -1.5), xycoords='data',
                xytext=((t_cut[slices_LP[0][0]]+t_cut[slices_LP[0][-1]])*1.9/5, -1.4), textcoords='data')


    ## Horizontal line and tag Burst PD
    plt.annotate("",
                xy=(t_cut[slices_PD[1][0]], -1.5), xycoords='data',
                xytext=(t_cut[slices_PD[1][-1]], -1.5), textcoords='data',
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))
    plt.annotate("Burst PD",
                xy=(t_cut[slices_PD[1][0]], -1.5), xycoords='data',
                xytext=((t_cut[slices_PD[1][0]]+t_cut[slices_PD[1][-1]])*2.25/5, -1.4), textcoords='data')


    ## Horizontal line and tag LP-PD delay
    plt.annotate("",
                xy=(t_cut[slices_PD[1][0]], -2), xycoords='data',
                xytext=(t_cut[slices_LP[0][-1]], -2), textcoords='data',
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))
    plt.annotate("LP-PD delay",
                xy=(t_cut[slices_PD[1][0]], -2), xycoords='data',
                xytext=((t_cut[slices_PD[1][0]]+t_cut[slices_LP[0][-1]])*2/5, -1.9), textcoords='data')


    ## Horizontal line and tag PD-LP delay
    plt.annotate("",
                xy=(t_cut[slices_LP[1][0]], -2), xycoords='data',
                xytext=(t_cut[slices_PD[1][-1]],-2), textcoords='data',
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))
    plt.annotate("PD-LP delay",
                xy=(t_cut[slices_LP[1][0]], -2), xycoords='data',
                xytext=((t_cut[slices_LP[1][0]]+t_cut[slices_PD[1][-1]])*2.4/5, -1.9), textcoords='data')
    

    ## Horizontal line and tag LP-PD interval
    plt.annotate("",
                xy=(t_cut[slices_LP[0][0]], -2.5), xycoords='data',
                xytext=(t_cut[slices_PD[1][0]], -2.5), textcoords='data',
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))
    plt.annotate("LP-PD interval",
                xy=(t_cut[slices_LP[0][0]], -2.5), xycoords='data',
                xytext=((t_cut[slices_LP[0][0]]+t_cut[slices_PD[1][0]])*2/5, -2.4), textcoords='data')
    

    ## Horizontal line and tag PD-LP interval
    plt.annotate("",
                xy=(t_cut[slices_PD[1][0]], -2.5), xycoords='data',
                xytext=(t_cut[slices_LP[1][0]], -2.5), textcoords='data',
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))
    plt.annotate("PD-LP interval",
                xy=(t_cut[slices_PD[1][0]], -2.5), xycoords='data',
                xytext=((t_cut[slices_PD[1][0]]+t_cut[slices_LP[1][0]])*2.35/5, -2.4), textcoords='data')


    plt.plot(t[x_min:x_max], V_LP[x_min:x_max]*factor_d2, color='lightseagreen')
    plt.plot(t[x_min:x_max], V_PD[x_min:x_max]+factor_d3*np.ones(np.size(V_PD[x_min:x_max])),color='salmon')
    plt.legend(['LP signal', 'PD signal'] )
    plt.ylim([-2.8,2.8])
    plt.xlabel('t (ms)')
    plt.ylabel('Voltaje (mV)')