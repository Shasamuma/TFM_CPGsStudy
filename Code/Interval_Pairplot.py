def Interval_Pairplot(Intervals, diagonal_kind='kde', Evolution=False, Palette="blend:salmon,lightseagreen", Cycle_by_cycle=True, hue="Normal", ishue2=False, hue2=None, diag='Normal', Palette2="blend:salmon,lightseagreen", color='#4ca89f'):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    if Evolution==False and Cycle_by_cycle==True:
        data_dict = {"Periodo LP (ms)": Intervals[:,0],
        "Periodo PD (ms)": Intervals[:,1],
        "Hiperpolarización LP (ms)": Intervals[:,2],
        "Hiperpolarización PD (ms)": Intervals[:,3],
        "Burst LP (ms)": Intervals[:,4],
        "Burst PD (ms)": Intervals[:,5],
        "Interval LPPD (ms)": Intervals[:,6],
        "Interval PDLP (ms)": Intervals[:,7],
        "Delay LPPD (ms)": Intervals[:,8],
        "Delay PDLP (ms)": Intervals[:,9]}

        data = pd.DataFrame(data_dict)
        sns.pairplot(data, diag_kind=diagonal_kind)

    elif Evolution==True and Cycle_by_cycle==True and not ishue2:
        if type(hue)==str:
            if hue=="Normal":
                Hue = range(len(Intervals[:,0]))
            elif hue=="Half":
                Hue = np.concatenate((np.zeros(int(len(Intervals[:,0])/2)), np.ones(int(len(Intervals[:,0])/2))))
        else:
            Hue = hue

        data_dict = {"Periodo LP (ms)": Intervals[:,0],
        "Periodo PD (ms)": Intervals[:,1],
        "Hiperpolarización LP (ms)": Intervals[:,2],
        "Hiperpolarización PD (ms)": Intervals[:,3],
        "Burst LP (ms)": Intervals[:,4],
        "Burst PD (ms)": Intervals[:,5],
        "Interval LPPD (ms)": Intervals[:,6],
        "Interval PDLP (ms)": Intervals[:,7],
        "Delay LPPD (ms)": Intervals[:,8],
        "Delay PDLP (ms)": Intervals[:,9],
        "hue": Hue}

        data = pd.DataFrame(data_dict)
        
        # Create a new figure to combine the elements
        fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(30, 30))

        # Upper triangle and diagonal
        for i in range(10):
            for j in range(10):
                if i<=j:
                    if i==j:  # Diagonal
                        sns.kdeplot(data, x=data.columns[i], ax=axes[i, j], fill=True, legend=False, color=color)
                    else:  # Lower triangle
                        sns.scatterplot(x=data.columns[j], y=data.columns[i], hue="hue", data=data, ax=axes[i, j], palette=Palette, legend=False)

        # Lower triangle
        for i in range(10):
            for j in range(10):
                if i>j:
                    sns.scatterplot(x=data.columns[j], y=data.columns[i], data=data, ax=axes[i, j], color=color, legend=False)
        # Share and eliminate axes        
        for i in range(10):
            for j in range(10):
                if j!=0 and i!=9:
                    if i!=j and j!=9 and i!=0:
                        axes[i,j].sharex(axes[j,j])
                        axes[i,j].sharey(axes[i,0])
                    elif i!=j and j==9:
                        if i!=0:
                            axes[i,j].sharey(axes[i,0])
                        elif i==0 and j!=1:
                            axes[i,j].sharey(axes[i,1])
                        axes[i,j].sharex(axes[j,j])
                    elif i!=j and j!=9 and i==0:
                        axes[i,j].sharex(axes[j,j])
                        if j!=1:
                            axes[i,j].sharey(axes[i,1])
                    elif i==j:
                        axes[i,j].sharex(axes[j,j])
                elif j==0:
                    if i==0:
                        axes[i,j].sharex(axes[j,j])
                    elif i!=0 and i!=9:
                        axes[i,j].sharex(axes[j,j])
                elif i==9 and j!=9:
                    axes[i,j].sharex(axes[j,j])
                    if j!=0:
                        axes[i,j].sharey(axes[i,0])

                axes[0,0].set(yticklabels=axes[0,1].get_yticklabels())
                axes[0,0].set(ylabel=axes[0,1].get_ylabel())

                if j!=0 and i!=9:
                    axes[i,j].get_yaxis().set_visible(False)
                    axes[i,j].get_xaxis().set_visible(False)

                elif j==0:
                    if i==0:
                        axes[i,j].get_xaxis().set_visible(False)
                    elif i!=0 and i!=9:
                        axes[i,j].get_xaxis().set_visible(False)
                elif j!=0 and i==9 and j!=9:
                    axes[i,j].get_yaxis().set_visible(False)
                elif j==9 and i==9:
                    axes[i,j].get_yaxis().set_visible(False)
        fig.tight_layout()
        plt.show()

    elif Evolution==True and Cycle_by_cycle==True and ishue2:
        if type(hue)==str:
            if hue=="Normal":
                Hue = range(len(Intervals[:,0]))
            elif hue=="Half":
                Hue = np.concatenate((np.zeros(int(len(Intervals[:,0])/2)), np.ones(int(len(Intervals[:,0])/2))))
        else:
            Hue = hue

        data_dict1 = {"Periodo LP (ms)": Intervals[:,0],
        "Periodo PD (ms)": Intervals[:,1],
        "Hiperpolarización LP (ms)": Intervals[:,2],
        "Hiperpolarización PD (ms)": Intervals[:,3],
        "Burst LP (ms)": Intervals[:,4],
        "Burst PD (ms)": Intervals[:,5],
        "Interval LPPD (ms)": Intervals[:,6],
        "Interval PDLP (ms)": Intervals[:,7],
        "Delay LPPD (ms)": Intervals[:,8],
        "Delay PDLP (ms)": Intervals[:,9],
        "hue": Hue}

        data_dict2 = {"Periodo LP (ms)": Intervals[:,0],
        "Periodo PD (ms)": Intervals[:,1],
        "Hiperpolarización LP (ms)": Intervals[:,2],
        "Hiperpolarización PD (ms)": Intervals[:,3],
        "Burst LP (ms)": Intervals[:,4],
        "Burst PD (ms)": Intervals[:,5],
        "Interval LPPD (ms)": Intervals[:,6],
        "Interval PDLP (ms)": Intervals[:,7],
        "Delay LPPD (ms)": Intervals[:,8],
        "Delay PDLP (ms)": Intervals[:,9],
        "hue2": hue2}

        data1 = pd.DataFrame(data_dict1)
        data2 = pd.DataFrame(data_dict2)
        
        # Create a new figure to combine the elements
        fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(30, 30))

        # Upper triangle and diagonal
        for i in range(10):
            for j in range(10):
                if i<=j:
                    if i==j:  # Diagonal
                        if diag=='Double':
                            sns.kdeplot(data2, x=data2.columns[i], ax=axes[i, j], fill=True, legend=False, hue='hue2', palette=Palette2)
                        else:
                            sns.kdeplot(data1, x=data1.columns[i], ax=axes[i, j], fill=True, legend=False, color=color)
                    else:  # Lower triangle
                        sns.scatterplot(x=data1.columns[j], y=data1.columns[i], hue="hue", data=data1, ax=axes[i, j], palette=Palette, legend=False)

        # Lower triangle
        for i in range(10):
            for j in range(10):
                if i>j:
                    sns.scatterplot(x=data2.columns[j], y=data2.columns[i], hue="hue2", data=data2, ax=axes[i, j], palette=Palette2, legend=False)
        # Share and eliminate axes        
        for i in range(10):
            for j in range(10):
                if j!=0 and i!=9:
                    if i!=j and j!=9 and i!=0:
                        axes[i,j].sharex(axes[j,j])
                        axes[i,j].sharey(axes[i,0])
                    elif i!=j and j==9:
                        if i!=0:
                            axes[i,j].sharey(axes[i,0])
                        elif i==0 and j!=1:
                            axes[i,j].sharey(axes[i,1])
                        axes[i,j].sharex(axes[j,j])
                    elif i!=j and j!=9 and i==0:
                        axes[i,j].sharex(axes[j,j])
                        if j!=1:
                            axes[i,j].sharey(axes[i,1])
                    elif i==j:
                        axes[i,j].sharex(axes[j,j])
                elif j==0:
                    if i==0:
                        axes[i,j].sharex(axes[j,j])
                    elif i!=0 and i!=9:
                        axes[i,j].sharex(axes[j,j])
                elif i==9 and j!=9:
                    axes[i,j].sharex(axes[j,j])
                    if j!=0:
                        axes[i,j].sharey(axes[i,0])

                axes[0,0].set(yticklabels=axes[0,1].get_yticklabels())
                axes[0,0].set(ylabel=axes[0,1].get_ylabel())

                if j!=0 and i!=9:
                    axes[i,j].get_yaxis().set_visible(False)
                    axes[i,j].get_xaxis().set_visible(False)

                elif j==0:
                    if i==0:
                        axes[i,j].get_xaxis().set_visible(False)
                    elif i!=0 and i!=9:
                        axes[i,j].get_xaxis().set_visible(False)
                elif j!=0 and i==9 and j!=9:
                    axes[i,j].get_yaxis().set_visible(False)
                elif j==9 and i==9:
                    axes[i,j].get_yaxis().set_visible(False)
        fig.tight_layout()
        plt.show()

    elif Evolution==False and Cycle_by_cycle==False:
        data_dict = {"Periodo LP (ms)": Intervals[:-1,0],
        "Periodo PD (ms)": Intervals[:-1,1],
        "Hiperpolarización LP (ms)": Intervals[:-1,2],
        "Hiperpolarización PD (ms)": Intervals[:-1,3],
        "Burst LP (ms)": Intervals[:-1,4],
        "Burst PD (ms)": Intervals[:-1,5],
        "Interval LPPD (ms)": Intervals[:-1,6],
        "Interval PDLP (ms)": Intervals[:-1,7],
        "Delay LPPD (ms)": Intervals[:-1,8],
        "Delay PDLP (ms)": Intervals[:-1,9],
        "Sig. Periodo LP (ms)": Intervals[1:,0],
        "Sig. Periodo PD (ms)": Intervals[1:,1],
        "Sig. Hiperpolarización LP (ms)": Intervals[1:,2],
        "Sig. Hiperpolarización PD (ms)": Intervals[1:,3],
        "Sig. Burst LP (ms)": Intervals[1:,4],
        "Sig. Burst PD (ms)": Intervals[1:,5],
        "Sig. Interval LPPD (ms)": Intervals[1:,6],
        "Sig. Interval PDLP (ms)": Intervals[1:,7],
        "Sig. Delay LPPD (ms)": Intervals[1:,8],
        "Sig. Delay PDLP (ms)": Intervals[1:,9]}

        data = pd.DataFrame(data_dict)
        # sns.pairplot(data, color=color, x_vars=["Periodo LP (ms)","Periodo PD (ms)","Hiperpolarización LP (ms)","Hiperpolarización PD (ms)","Burst LP (ms)","Burst PD (ms)","Interval LPPD (ms)","Interval PDLP (ms)","Delay LPPD (ms)","Delay PDLP (ms)"],
        #      y_vars=["Sig. Periodo LP (ms)","Sig. Periodo PD (ms)","Sig. Hiperpolarización LP (ms)","Sig. Hiperpolarización PD (ms)","Sig. Burst LP (ms)","Sig. Burst PD (ms)","Sig. Interval LPPD (ms)","Sig. Interval PDLP (ms)","Sig. Delay LPPD (ms)","Sig. Delay PDLP (ms)"])
        
        fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(30, 30))

        # Upper triangle and diagonal
        for i in range(10):
            for j in range(10):
                if i<=j:
                    if i==j:  # Diagonal
                        sns.kdeplot(data, x=data.columns[i], ax=axes[i, j], fill=True, legend=False, color=color)
                    else:  # Lower triangle
                        sns.scatterplot(x=data.columns[j], y=data.columns[i+10], data=data, color=color,ax=axes[i, j], legend=False)
                if i>j:
                    sns.scatterplot(x=data.columns[j], y=data.columns[i+10], data=data, ax=axes[i, j], color=color, legend=False)

        # Share and eliminate axes        
        for i in range(10):
            for j in range(10):
                if j!=0 and i!=9:
                    if i!=j and j!=9 and i!=0:
                        axes[i,j].sharex(axes[j,j])
                        axes[i,j].sharey(axes[i,0])
                    elif i!=j and j==9:
                        if i!=0:
                            axes[i,j].sharey(axes[i,0])
                        elif i==0 and j!=1:
                            axes[i,j].sharey(axes[i,1])
                        axes[i,j].sharex(axes[j,j])
                    elif i!=j and j!=9 and i==0:
                        axes[i,j].sharex(axes[j,j])
                        if j!=1:
                            axes[i,j].sharey(axes[i,1])
                    elif i==j:
                        axes[i,j].sharex(axes[j,j])
                elif j==0:
                    if i==0:
                        axes[i,j].sharex(axes[j,j])
                    elif i!=0 and i!=9:
                        axes[i,j].sharex(axes[j,j])
                elif i==9 and j!=9:
                    axes[i,j].sharex(axes[j,j])
                    if j!=0:
                        axes[i,j].sharey(axes[i,0])

                axes[0,0].set(yticklabels=axes[0,1].get_yticklabels())
                axes[0,0].set(ylabel=axes[0,1].get_ylabel())

                if j!=0 and i!=9:
                    axes[i,j].get_yaxis().set_visible(False)
                    axes[i,j].get_xaxis().set_visible(False)

                elif j==0:
                    if i==0:
                        axes[i,j].get_xaxis().set_visible(False)
                    elif i!=0 and i!=9:
                        axes[i,j].get_xaxis().set_visible(False)
                elif j!=0 and i==9 and j!=9:
                    axes[i,j].get_yaxis().set_visible(False)
                elif j==9 and i==9:
                    axes[i,j].get_yaxis().set_visible(False)
        fig.tight_layout()
        plt.show()
    else:
        print('Error: the possible combinations are: ')
        print('1: Evolution=False and Cycle_by_cycle=True')
        print('2: Evolution=True and Cycle_by_cycle=True')
        print('3: Evolution=False and Cycle_by_cycle=False')
        print('4: Evolution=True and Cycle_by_cycle=True and ishue2!=None')

def KDE_plot(Intervals, Hue, hue2, Palette="blend:salmon,lightseagreen", Palette2="blend:salmon,lightseagreen", color='#4ca89f'):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    data_dict1 = {"Periodo LP (ms)": Intervals[:,0],
        "Periodo PD (ms)": Intervals[:,1],
        "Hiperpolarización LP (ms)": Intervals[:,2],
        "Hiperpolarización PD (ms)": Intervals[:,3],
        "Burst LP (ms)": Intervals[:,4],
        "Burst PD (ms)": Intervals[:,5],
        "Interval LPPD (ms)": Intervals[:,6],
        "Interval PDLP (ms)": Intervals[:,7],
        "Delay LPPD (ms)": Intervals[:,8],
        "Delay PDLP (ms)": Intervals[:,9],
        "hue": Hue}

    data_dict2 = {"Periodo LP (ms)": Intervals[:,0],
        "Periodo PD (ms)": Intervals[:,1],
        "Hiperpolarización LP (ms)": Intervals[:,2],
        "Hiperpolarización PD (ms)": Intervals[:,3],
        "Burst LP (ms)": Intervals[:,4],
        "Burst PD (ms)": Intervals[:,5],
        "Interval LPPD (ms)": Intervals[:,6],
        "Interval PDLP (ms)": Intervals[:,7],
        "Delay LPPD (ms)": Intervals[:,8],
        "Delay PDLP (ms)": Intervals[:,9],
        "hue2": hue2}
    
    data1 = pd.DataFrame(data_dict1)
    data2 = pd.DataFrame(data_dict2)
    

    fig, axes = plt.subplots(nrows=3, ncols=10, figsize=(30, 10))
    for j in range(10):
        sns.kdeplot(data1, x=data2.columns[j], ax=axes[0, j], fill=True, legend=False, color=color)
        sns.kdeplot(data1, x=data2.columns[j], ax=axes[1, j], fill=True, legend=False, hue='hue', palette=Palette)
        sns.kdeplot(data2, x=data2.columns[j], ax=axes[2, j], fill=True, legend=False, hue='hue2', palette=Palette2)
    fig.tight_layout()
    plt.show()

    



        