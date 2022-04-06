import pandas as pd
import matplotlib.pyplot as plt


def pos_or_neg_skewed(val):
    if val >= 0:
        print("Positively Skewed\n\n")
    else:
        print("Negatively Skewed\n\n")


def num_of_outliers(data, upper, lower):
    outliers = []
    for i in data:
        if i >= upper or i <= lower:
            outliers.append(i)
    return outliers


filename = pd.read_csv('Netflix subscription fee Dec-2021.csv')

cost = filename['Cost Per Month - Basic ($)']
movies = filename['No. of Movies']
tv_shows = filename['No. of TV Shows']
lib_size = filename['Total Library Size']

plt.scatter(movies, lib_size, edgecolors='r')
plt.xlabel('Movies')
plt.ylabel('Library Size')
plt.title('Movie Library Size')
plt.show()

plt.scatter(tv_shows, movies, edgecolors='r')
plt.xlabel('TV Shows')
plt.ylabel('Movies')
plt.title('Movies & TV Shows')
plt.show()

for col in filename.columns:

    if filename[col].dtype != object:

        Q1 = filename[col].quantile(0.25)
        Q3 = filename[col].quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + 1.5 * IQR
        lower = Q1 - 1.5 * IQR

        print(col)
        print("IQR: ", IQR)
        print("Mean: ", filename[col].mean())
        print("Median: ", filename[col].median())
        print("Mode: ", filename[col].mode().values)
        print("Max: ", filename[col].max())
        print("Min: ", filename[col].min())

        print("Outliers(Upper): ", upper)
        print("Outliers(Lower): ", lower)
        print("List of outliers: ", num_of_outliers(filename[col], upper, lower))
        pos_or_neg_skewed(filename[col].skew())

        filename.boxplot(col)
        plt.show()

        new_filename = filename[filename[col] < upper]
        new_filename.shape

        new_filename = filename[filename[col] > lower]
        new_filename.shape

        new_filename.boxplot(col)
        plt.show()
