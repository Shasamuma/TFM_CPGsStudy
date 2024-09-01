def Plot_Data(Data):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    data = pd.read_csv(Data, delimiter='\t', header=None, decimal=',').dropna(axis=1)

    ## Time
    t = data.take([0], axis=1)
    t = t.to_numpy().reshape(np.shape(t.to_numpy())[0])
    t = t - t[0] * np.ones(np.shape(t))
    dt = t[1]-t[0]
    
    ## LP signal
    V_LP = data.take([1], axis=1)
    V_LP = V_LP.to_numpy().reshape(np.shape(V_LP.to_numpy())[0])
    
    ## PD signal
    V_PD = data.take([2], axis=1)
    V_PD = V_PD.to_numpy().reshape(np.shape(V_PD.to_numpy())[0])

    ## Plot data
    x_min = 0
    x_max = 55000
    fig, ax = plt.subplots(2,1, figsize=(10, 10))
    ax2 = ax[0].twiny()
    ax2.plot(np.arange(x_min, x_max+1,1), np.zeros(x_max+1), color='White', alpha=0.01) 
    
    ax[0].plot(t[x_min:x_max], V_LP[x_min:x_max], color='lightseagreen')
    ax[0].plot(t[x_min:x_max], V_PD[x_min:x_max], color='salmon')
    ax[0].legend(['LP signal', 'PD signal'])
    ax[0].set_xlabel('t (ms)')
    ax[0].set_ylabel('Voltaje (mV)')
    ax[0].tick_params(labelsize='small')
    ax[0].set_xticks(np.arange(0, t[x_max]+1, step=int(t[x_max]/15)))
    ax[0].grid(linestyle='--', linewidth=0.5)
    
    x_min = 0
    x_max = 10000
    ax2 = ax[1].twiny()
    ax2.plot(np.arange(x_min, x_max+1,1), np.zeros(x_max+1), color='White', alpha=0.01) 
    ax[1].plot(t[x_min:x_max], V_LP[x_min:x_max], color='lightseagreen')
    ax[1].plot(t[x_min:x_max], V_PD[x_min:x_max], color='salmon')
    ax[1].legend(['LP signal', 'PD signal'])
    ax[1].set_xlabel('t (ms)')
    ax[1].set_ylabel('Voltaje (mV)')
    ax[1].tick_params(labelsize='small')
    ax[1].set_xticks(np.arange(0, t[x_max]+1, step=int(t[x_max])/18))
    ax[1].grid(linestyle='--', linewidth=0.5)
    plt.show()
    

def Plot_Data_2(Data):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    data = pd.read_csv(Data, delimiter='\t', header=None, decimal=',').dropna(axis=1)

    ## Time
    t = data.take([0], axis=1)
    t = t.to_numpy().reshape(np.shape(t.to_numpy())[0])
    t = t - t[0] * np.ones(np.shape(t))
    dt = t[1]-t[0]
    
    ## LP signal
    V_LP = data.take([1], axis=1)
    V_LP = V_LP.to_numpy().reshape(np.shape(V_LP.to_numpy())[0])
    
    ## PD signal
    V_PD = data.take([2], axis=1)
    V_PD = V_PD.to_numpy().reshape(np.shape(V_PD.to_numpy())[0])

    ## Plot data
    x_min = 1800
    x_max = 9500
    fig, ax = plt.subplots(2,1, figsize=(10, 7), sharex=True)

    ax[0].plot(t[x_min:x_max], V_LP[x_min:x_max], color='black')
    ax[0].plot(t[3900+1000:6700-500], V_LP[3900+1000:6700-500], color='lightseagreen')
    ax[0].plot(t[2100:3900], V_LP[2100:3900], color='salmon')
    ax[0].plot(t[6700:x_max-600], V_LP[6700:x_max-600], color='saddlebrown')
    ax[0].legend(['Extracellular recording','LP signal', 'PD signal', 'PY signal'])
    ax[0].set_ylabel('Voltaje (mV)')
    ax[0].tick_params(labelsize='small')
    ax[0].grid(linestyle='--', linewidth=0.5)
    
    ax[1].plot(t[x_min:x_max], V_PD[x_min:x_max], color='salmon')
    ax[1].legend(['PD signal'])
    ax[1].set_xlabel('t (ms)')
    ax[1].set_ylabel('Voltaje (mV)')
    ax[1].tick_params(labelsize='small')
    ax[1].grid(linestyle='--', linewidth=0.5)
    plt.show()
    fig.savefig('./AQUI.jpg')