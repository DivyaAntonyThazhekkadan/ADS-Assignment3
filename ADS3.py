# -*- coding: utf-8 -*-
"""
Created on Tue May  9 22:41:44 2023

@author: user
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def read_data(file_path):
    """Read data from a file into a DataFrame."""
    data = pd.read_excel(file_path)
    data = data.drop(['Country Code', 'Indicator Code'], axis=1)
    data = data.drop_duplicates()
    columnlist = ['Country Name', 'Indicator Name', '1990',	'1991',	'1992',	'1993', '1994',	'1995',	'1996',	'1997', '1998',	'1999',	'2000',	'2001',
                  '2002',	'2003',	'2004',	'2005',	'2006',	'2007',	'2008',	'2009', '2010',	'2011',	'2012',	'2013',	'2014',	'2015',	'2016',	'2017', '2018',	'2019']
    # data = data.dropna()
    data = data[columnlist]
    data = data.fillna(0)
    return data


def normalize_data(data):
    """Normalize data by subtracting the mean and dividing by the standard deviation."""
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    return data_normalized


def run_kmeans(data, n_clusters):
    """Run k-means clustering on the data."""
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    return kmeans.labels_


def plot_clusters(data, labels, centers):
    """Plot the data with cluster labels and cluster centers."""
    colors = plt.cm.Spectral(labels.astype(float) / len(centers))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=colors)
    center_colors = plt.cm.Spectral(
        np.arange(len(centers)).astype(float) / len(centers))
    plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=200, linewidths=3,
                color=center_colors, zorder=10)
    for i, txt in enumerate(data.index):
        plt.annotate(txt, (data.iloc[i, 0], data.iloc[i, 1]))

    plt.xlabel(data.columns[0][1])
    plt.ylabel(data.columns[1][1])
    plt.title('Cluster Plot')
    plt.ylim([min(data.iloc[:, 1])-3, max(data.iloc[:, 1])+1])

    plt.savefig('cluster.png', bbox_inches="tight")
    plt.show()


def func(x, a, b, c):
    return a * np.exp(-b * x) + c


# Read the data from a file
data = read_data('API_19_DS2_en_excel_v2_5360124.xlsx')

# Filter the data to only include specific countries and indicators
countries = ['United States', 'China', 'India',
             'Brazil', 'Germany', 'South Africa']
indicators = ['CO2 emissions (metric tons per capita)',
              'Renewable energy consumption (% of total final energy consumption)']
data = data[data['Country Name'].isin(countries)]
data = data[data['Indicator Name'].isin(indicators)]
do = data
# Reshape the data to have one row for each country and one column for each year
data = data.pivot_table(
    index='Country Name', columns='Indicator Name').reset_index()
data.set_index('Country Name')

# Set the index of the data DataFrame to the 'Country Name' column
data = data.set_index('Country Name')
dp = data
# Normalize the data
data_normalized = normalize_data(data.iloc[:, 1:])

# Run k-means clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(data_normalized)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Plot the clusters
plot_clusters(data.iloc[:, 1:], labels, centers)


# fitted models

# Create a figure with 2 rows and 3 columns
fig, axs = plt.subplots(2, 3)

# Fit a model for each cluster and each indicator
for j in range(2):
    # Select the data for the current indicator
    indicator = indicators[j]
    data = do[do['Indicator Name'] == indicator]

    # Use the existing cluster labels
    labels = labels

    for i in range(3):
        # Select one country from the cluster
        country = data.iloc[np.where(labels == i)[0][0], 0]

        # Extract the x and y values from the data
        xdata = data.iloc[:, 2:].columns.astype(int)
        ydata = data[data['Country Name'] == country].iloc[:, 2:].values[0]

        # Fit the exponential growth model to the data
        popt, pcov = curve_fit(func, xdata, ydata)

        # Make predictions for the next 10 years
        xpred = np.arange(xdata[-1]+1, xdata[-1]+11)
        ypred = func(xpred, *popt)

        # Plot the data and the fitted model
        axs[j, i].plot(xdata, ydata, 'ko', label="Original Data")
        axs[j, i].plot(xdata, func(xdata, *popt), 'r-', label="Fitted Curve")
        axs[j, i].plot(xpred, ypred, 'b--', label="Predictions")
        axs[j, i].set_title(f'Cluster {i+1}: {country}')
        axs[j, i].legend()

plt.show()
