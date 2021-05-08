import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, SelectKBest


class Preparation:
    def __init__(self, path, format):
        self.path = path
        self.format = format

    def Preparation(self):
        try:
            data = pd.read_csv(self.path, sep=';')
            Y = data['quality']
            X = data.drop('quality', axis=1)
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.35, random_state=150)

            # Features Selection (based on mutual information gain)
            mutual_info = mutual_info_classif(x_train, y_train)
            # print(mutual_info)

            mutual_info = pd.Series(mutual_info)
            mutual_info.index = x_train.columns
            # print(mutual_info.sort_values(ascending=False))
            # mutual_info.sort_values(ascending=False).plot.bar(figsize=(10, 5))
            # plt.show()

            sel_six_cols = SelectKBest(mutual_info_classif, k=6)
            sel_six_cols.fit(x_train, y_train)
            sel_cols = x_train.columns[sel_six_cols.get_support()]
            # print("sel_seven_cols:  ", sel_cols)
            selected_columns = [col for col in sel_cols]
            # ('fixed acidity', 'citric acid', 'chlorides', 'pH', 'sulphates', 'alcohol')

            # now select those features
            x_train = x_train[[col for col in selected_columns]]
            x_test = x_test[[col for col in selected_columns]]
            # print(x_test.iloc[9])

            # Standarized the x_train and x_test data
            scaler = StandardScaler()
            x_scaled_train = scaler.fit_transform(x_train)
            x_scaled_test = scaler.fit_transform(x_test)
            return x_scaled_train, x_scaled_test, y_train, y_test, selected_columns

        except Exception as e:
            return e


if __name__ == "__main__":
    path1 = "..\Data\Prepared Data\winequality-white.csv"
    path2 = "..\Data\Prepared Data\winequality-red.csv"
    f = "csv"
    pp = Preparation(path1, f)
    pp.Preparation()
