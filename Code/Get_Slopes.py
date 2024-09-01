def Get_Slopes(Intervals, threshold=0.85):
    import numpy as np
    from scipy.stats import linregress
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    N = Intervals.shape[1]

    data = pd.DataFrame({"Periodo LP": Intervals[:,0],
        "Periodo PD": Intervals[:,1],
        "Hiperpolarización LP": Intervals[:,2],
        "Hiperpolarización PD": Intervals[:,3],
        "Burst LP": Intervals[:,4],
        "Burst PD": Intervals[:,5],
        "Interval LPPD": Intervals[:,6],
        "Interval PDLP": Intervals[:,7],
        "Delay LPPD": Intervals[:,8],
        "Delay PDLP": Intervals[:,9]})

    ## Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    ## Fit and transform the data (normalize each column independently)
    norm_data = scaler.fit_transform(data)
    norm_data = pd.DataFrame(norm_data, columns=data.columns)

    ## Initialize results 
    results = []

    for i in range(N):
        for j in range(i+1,N):
            x = norm_data.iloc[:, j]
            y = norm_data.iloc[:, i]

            slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
            ## Calculate R^2
            r_squared = r_value ** 2
            
            ## Determine if the dataset follows a linear trend
            is_linear = r_squared > threshold  # Example threshold for R^2
            if is_linear==True:
                x = data.iloc[:, j]
                y = data.iloc[:, i]

                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                ## Calculate R^2
                r_squared = r_value ** 2
            
                ## Determine if the dataset follows a linear trend
                is_linear = r_squared > 0.85  # Example threshold for R^2
            
                ## Store the results
                results.append({
                    'Dataset': ' + '.join([data.columns[j], data.columns[i]]),
                    'Slope': slope,
                    'Intercept': intercept,
                    'R2': r_squared,
                    'Linear?': is_linear})
    results_df = pd.DataFrame(results)

    return results_df


