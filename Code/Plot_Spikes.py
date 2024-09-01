def Plot_Spikes_Offline(t, V_PD, V_LP, slices_PD, slices_LP, xmin, xmax):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    inis_PD = [i[0] for i in slices_PD if i[0]<xmax and i[0]>=xmin]
    finis_PD = [i[-1] for i in slices_PD if i[-1]<xmax and i[-1]>=xmin]

    inis_LP = [i[0] for i in slices_LP if i[0]<xmax and i[0]>=xmin]
    finis_LP = [i[-1] for i in slices_LP if i[-1]<xmax and i[-1]>=xmin]
    
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(t[xmin:xmax], V_LP[xmin:xmax], color='lightseagreen')
    ax.plot(t[xmin:xmax], V_PD[xmin:xmax], color='salmon')
    ax.plot(t[inis_LP+finis_LP], V_LP[inis_LP+finis_LP], color='darkslategray', marker='o', linestyle='')
    ax.plot(t[inis_PD+finis_PD], V_PD[inis_PD+finis_PD], color='firebrick', marker='o', linestyle='')
    plt.legend(['LP signal', 'PD signal'] )
    plt.xlabel('t (ms)')
    plt.ylabel('Voltaje (mV)')
    
    
    
    
def Plot_Spikes_Online(t, V_PD, V_LP, t_PD, t_LP, xmin, xmax):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    inis_PD = [i for i in t_PD[:,0] if i<t[xmax] and i>=t[xmin]]
    finis_PD = [i for i in t_PD[:,1] if i<t[xmax] and i>=t[xmin]]

    inis_LP = [i for i in t_LP[:,0] if i<t[xmax] and i>=t[xmin]]
    finis_LP = [i for i in t_LP[:,1] if i<t[xmax] and i>=t[xmin]]

    idx_inis_LP = [np.argmin(np.abs(t - i*np.ones(np.shape(t)))) for i in inis_LP]
    idx_finis_LP = [np.argmin(np.abs(t - i*np.ones(np.shape(t)))) for i in finis_LP]
    idx_inis_PD = [np.argmin(np.abs(t - i*np.ones(np.shape(t)))) for i in inis_PD]
    idx_finis_PD = [np.argmin(np.abs(t - i*np.ones(np.shape(t)))) for i in finis_PD]

    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(t[xmin:xmax], V_LP[xmin:xmax], color='lightseagreen')
    ax.plot(t[xmin:xmax], V_PD[xmin:xmax], color='salmon')
    ax.plot(inis_LP, V_LP[idx_inis_LP], color='darkslategray', marker='o', linestyle='')
    ax.plot(finis_LP, V_LP[idx_finis_LP], color='firebrick', marker='o', linestyle='')
    ax.plot(inis_PD, V_PD[idx_inis_PD], color='firebrick', marker='o', linestyle='')
    ax.plot(finis_PD, V_PD[idx_finis_PD], color='darkslategray', marker='o', linestyle='')
    plt.legend(['LP signal', 'PD signal'] )
    plt.xlabel('t (ms)')
    plt.ylabel('Voltaje (mV)')