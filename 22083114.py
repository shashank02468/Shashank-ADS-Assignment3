#Required Libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def dataFrame_loading_cleaning(filePath):
    """
        Load and clean a dataset from a CSV file.

        Parameters:
        - filePath (str): The path to the CSV file.

        Returns:
        - data_frame (pd.DataFrame): The original DataFrame
        loaded from the CSV file.
        - clean_data_frame (pd.DataFrame): The cleaned DataFrame
         with NaN values removed.
        - transposed_clean_data_frame (pd.DataFrame): The transposed
         and cleaned DataFrame.
        """
    #load Dataset
    data_frame = pd.read_csv(filePath)
    #cleaning Dataset
    clean_data_frame = data_frame.dropna()
    #cleaned and transposed dataset
    tranposed_clean_data_frame = clean_data_frame.transpose()

    return data_frame , clean_data_frame , tranposed_clean_data_frame


def clusterPlot(data):
    """
       Plot a scatter plot of data points with cluster
       colors and cluster centers.

       Parameters:
       - data_selected (pd.DataFrame): The DataFrame containing
       selected columns for plotting.
       - scaler (StandardScaler): The StandardScaler used
       for normalization.
       - kmeans (KMeans): The KMeans clustering model.

       Returns:
       None
       """
    plt.figure(figsize = (12 , 8))

    # Scatter plot of data points
    plt.scatter(data_selected['CO2 emissions (kg per 2015 US$ of GDP) [EN.ATM.CO2E.KD.GD]'] ,
                data_selected['Urban population [SP.URB.TOTL]'] ,
                c = data['Cluster'] , cmap = 'viridis' , s = 50 , alpha = 0.8)

    # Plot cluster centers
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    plt.scatter(centers[: , 3] , centers[: , 4] , c = 'red' ,
                marker = 'X' , s = 200 , label = 'Cluster Centers')

    plt.title('Clustering of Countries' , fontsize = 14)
    plt.xlabel('CO2 emissions (kg per 2015 US$ of GDP)' , fontsize = 14)
    plt.ylabel('Urban population' , fontsize = 14)
    plt.legend()
    plt.show()


def simple_model(x , a , b):
    """
        Simple linear model function.

        Parameters:
        - x (array-like): Independent variable values.
        - a (float): Slope coefficient.
        - b (float): Intercept coefficient.

        Returns:
        - array-like: Predicted values based on the simple linear model.
        """
    return a * x + b


def curveFitPlot():
    """
        Plot the data with the fitted curve and confidence range.

        Parameters:
        None

        Returns:
        None
        """
    # Plotting the data and the fitted curve with confidence range
    plt.scatter(x_data , y_data , label = 'Data')

    # Extend x_range to include 2023
    x_range = np.linspace(min(x_data) , 2023 , 100)

    # Calculate y-values using the fitted parameters
    y_fit = simple_model(x_range , *popt)
    y_err = np.sqrt(np.diag(pcov))

    # Repeat the y_err array to match the length of y_fit
    y_err = np.repeat(y_err[: , np.newaxis] , len(x_range) , axis=1)

    y_upper = y_fit + y_err
    y_lower = y_fit - y_err


    plt.fill_between(x_range , y_upper[0 , :] , y_lower[0 , :] ,
                     color = 'black' , alpha = 0.2 , label = 'Confidence Range')
    plt.plot(x_range , y_fit , label = 'Best Fit' , color = 'black')
    year_2023 = 2023

    # Convert 'Urban population [SP.URB.TOTL]' column to numeric
    org_df['Urban population [SP.URB.TOTL]'] = pd.to_numeric(org_df['Urban population [SP.URB.TOTL]'] ,
                                                             errors = 'coerce')

    # Use the simple_model function with the parameters obtained from curve fitting
    org_df['Predicted_CO2_2023'] = simple_model(year_2023 , *popt) * \
                                   org_df['Urban population [SP.URB.TOTL]']

    # Display the results in a table format
    predicted_results = org_df[['Country Name' , 'Predicted_CO2_2023']]
    print(predicted_results.head(13))

    plt.xlabel('Time' , fontsize = 14)
    plt.ylabel('CO2 emissions (kg per 2015 US$ of GDP)' , fontsize = 14)
    plt.legend()
    plt.show()


#Load,clean,transposed dataframe
org_df , cleaned_df , tranpose_df = dataFrame_loading_cleaning('sashank.csv')

print('original dataFrame')
print(org_df)
print('cleaned dataframe')
print(cleaned_df)
print('transposed dataframe')
print(tranpose_df)
required_columns = ["Forest area (% of land area) [AG.LND.FRST.ZS]" ,
                    "Access to electricity, rural (% of rural population) [EG.ELC.ACCS.RU.ZS]" ,
                    "Agricultural land (% of land area) [AG.LND.AGRI.ZS]" ,
                    "CO2 emissions (kg per 2015 US$ of GDP) [EN.ATM.CO2E.KD.GD]" ,
                    "Urban population [SP.URB.TOTL]"]

# convert string columns to numeric columns
data_selected = org_df[required_columns]
data_selected = data_selected.apply(pd.to_numeric , errors = 'coerce')

# Handle missing values (replace ".." with NaN and then fill NaN values)
data_selected.replace(".." , np.nan , inplace = True)
data_selected.fillna(data_selected.mean() , inplace = True)

#Normalization
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data_selected)

# Clustering
kmeans = KMeans(n_clusters = 5 , random_state = 42)
org_df['Cluster'] = kmeans.fit_predict(data_normalized)

# Compute and print the silhouette score
silhouette_avg = silhouette_score(data_normalized , org_df['Cluster'])
print(f"Silhouette Score: {silhouette_avg}")

clusterPlot(org_df)

data = org_df.apply(pd.to_numeric , errors = 'coerce')
data.replace(".." , np.nan , inplace = True)
data.fillna(data.mean() , inplace = True)

x_data = data['Time'].values
y_data = data['CO2 emissions (kg per 2015 US$ of GDP) [EN.ATM.CO2E.KD.GD]'].values

# Filter out NaN values
valid_data = ~np.isnan(x_data) & ~np.isnan(y_data)
x_data = x_data[valid_data]
y_data = y_data[valid_data]

popt , pcov = curve_fit(simple_model , x_data , y_data)
print(popt , pcov)
curveFitPlot()


