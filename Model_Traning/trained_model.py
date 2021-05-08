import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, minmax_scale
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from Data_Preparation import prepare


class Model_Training:
    def __init__(self, path, format):
        self.path = path
        self.format = format

    # Using Logistic Regression Algorithm
    def LogisticRegression(self):
        try:
            os = prepare.Preparation(self.path, self.format)
            x_train, x_test, y_train, y_test, columns = os.Preparation()

            # fit based on Standarized data
            logistic = LogisticRegression(random_state=175)
            logistic.fit(x_train, y_train)

            # check the accuracy score based on train data
            y_predict1 = logistic.predict(x_train)
            logistic_accuracy1 = accuracy_score(y_train, y_predict1)
            print("Accuracy_score Train(logistic) : ", logistic_accuracy1)

            # check the accuracy score based on test data
            y_predict2 = logistic.predict(x_test)
            logistic_accuracy2 = accuracy_score(y_test, y_predict2)
            print("Accuracy_score Test(logistic) : ", logistic_accuracy2, '\n')
            return logistic_accuracy2, logistic, columns

        except Exception as e:
            return e

    # Using Decision Tree Classifier Algorithm
    def DecisionTreeClassifier(self):
        try:
            os = prepare.Preparation(self.path, self.format)
            x_train, x_test, y_train, y_test, columns = os.Preparation()

            # Apply Decision Tree
            decesion1 = DecisionTreeClassifier(random_state=175)
            decesion1.fit(x_train, y_train)

            # check the accuracy score based on train data
            y_predict01 = decesion1.predict(x_train)
            decision_accuracy01 = accuracy_score(y_train, y_predict01)
            print("Accuracy_score Train(decesion1): ", decision_accuracy01)

            # check the accuracy score based on test data
            y_predict02 = decesion1.predict(x_test)
            decision_accuracy02 = accuracy_score(y_test, y_predict02)
            print("Accuracy_score Test(decesion1): ", decision_accuracy02, '\n')

            # Try to improve the accuracy score
            # Apply GridSearch
            decision = DecisionTreeClassifier()
            grid_param = {
                'criterion': ['gini', 'entropy'],
                'max_depth': range(2, 32, 1),
                'min_samples_leaf': range(1, 10, 1),
                'min_samples_split': range(2, 10, 1),
                'splitter': ['best', 'random']
            }
            grid_search = GridSearchCV(estimator=decision,
                                       param_grid=grid_param,
                                       cv=5,
                                       n_jobs=-1)
            # grid_search.fit(x_train, y_train)
            # best_parameters = grid_search.best_params_
            # print(best_parameters)
            # criterion, max_depth, min_samples_leaf, min_samples_split, splitter = tuple(value for key, value in best_parameters.items())
            dct = {'criterion': 'gini', 'max_depth': 22, 'min_samples_leaf': 1, 'min_samples_split': 4,
                   'splitter': 'best'}
            criterion, max_depth, min_samples_leaf, min_samples_split, splitter = tuple(
                value for key, value in dct.items())

            # fit the value in DecisionTreeClassifier
            decisions = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                               min_samples_leaf=min_samples_leaf,
                                               min_samples_split=min_samples_split, splitter=splitter, random_state=175)
            decisions.fit(x_train, y_train)

            # check the accuracy score based on train data
            y_predict1 = decisions.predict(x_train)
            decision_accuracy1 = accuracy_score(y_train, y_predict1)
            print("Accuracy_score Train(decesion2): ", decision_accuracy1)

            # check the accuracy score based on test data
            y_predict2 = decisions.predict(x_test)
            decision_accuracy2 = accuracy_score(y_test, y_predict2)
            print("Accuracy_score Test(decesion2): ", decision_accuracy2, '\n')
            # print(y_predict2)

            # Improve the accuracy score
            bagg_dec = BaggingClassifier(decisions, n_estimators=20, max_samples=0.5,
                                         bootstrap=True, random_state=175, oob_score=True)
            bagg_dec.fit(x_train, y_train)

            # Claculate accuracy score
            decision_accuracy11 = bagg_dec.score(x_train, y_train)
            decision_accuracy12 = bagg_dec.score(x_test, y_test)
            print("Based on train data(decesion_bagg): ", decision_accuracy11)
            print("Based on test data(decesion_bagg): ", decision_accuracy12, '\n')
            # print(bagg_dec.predict(x_test))

            # compare all accuracy (decision_accuracy02, decision_accuracy2, decision_accuracy12)
            if (decision_accuracy02 > decision_accuracy2) and (decision_accuracy02 > decision_accuracy12):
                return decision_accuracy02, decesion1, columns
            elif (decision_accuracy2 > decision_accuracy02) and (decision_accuracy2 > decision_accuracy12):
                return decision_accuracy2, decisions, columns
            else:
                return decision_accuracy12, bagg_dec, columns

        except Exception as e:
            return e

    # Using Random Forest Classifier Algorithm
    def RandomForestClassifier(self):
        try:
            os = prepare.Preparation(self.path, self.format)
            x_train, x_test, y_train, y_test, columns = os.Preparation()

            # Apply Random Forest
            random1 = RandomForestClassifier(random_state=175, n_estimators=50)
            random1.fit(x_train, y_train)

            # Check accuracy based on train data
            y_predict1 = random1.predict(x_train)
            accuracy1 = accuracy_score(y_train, y_predict1)
            print("Accuracy based on train data(random1): ", accuracy1)

            # Check accuracy based on test data
            y_predict2 = random1.predict(x_test)
            accuracy2 = accuracy_score(y_test, y_predict2)
            print("Accuracy based on train data(random1): ", accuracy2, '\n')
            # print(y_predict2[0:50])

            # Imporove the accuracy score, so that model is less overfitted model
            # Apply GridSearch
            grid_param = {
                "n_estimators": [90, 100, 130, 160, 190],
                'criterion': ['gini', 'entropy'],
                'max_depth': range(2, 20, 1),
                'min_samples_leaf': range(1, 10, 1),
                'min_samples_split': range(2, 10, 1),
                'max_features': ['auto', 'log2']
            }
            grid_search = GridSearchCV(estimator=random1, param_grid=grid_param, cv=5,
                                       n_jobs=-1, verbose=3)
            # grid_search.fit(x_train, y_train)
            # best_params = grid_search.best_params_
            # print(best_params)
            # criterion, max_depth, max_features, min_samples_leaf, min_samples_split, n_estimators = tuple(value for key, value in best_params.items())

            dct = {'criterion': 'gini', 'max_depth': 12, 'max_features': 'auto',
                   'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 130}
            criterion, max_depth, max_features, min_samples_leaf, min_samples_split, n_estimators = tuple(
                value for key, value in dct.items())
            random = RandomForestClassifier(criterion=criterion,
                                            max_depth=max_depth,
                                            max_features=max_features,
                                            min_samples_leaf=min_samples_leaf,
                                            min_samples_split=min_samples_split,
                                            n_estimators=n_estimators, random_state=175)
            random.fit(x_train, y_train)
            # Check accuracy based on train data
            y_predict11 = random.predict(x_train)
            accuracy11 = accuracy_score(y_train, y_predict11)
            print("Accuracy based on train data(random2): ", accuracy1)

            # Check accuracy based on test data
            y_predict22 = random.predict(x_test)
            accuracy22 = accuracy_score(y_test, y_predict22)
            print("Accuracy based on train data(random2): ", accuracy22, '\n')

            # Compare based on accuracy (accuracy2, accuracy22)
            if accuracy2 > accuracy22:
                return accuracy2, random1, columns
            else:
                return accuracy22, random, columns

        except Exception as e:
            return e

    # Using XGBoost Classifier Algorithm
    def GradientBoostingClassifier(self):
        try:
            os = prepare.Preparation(self.path, self.format)
            x_train, x_test, y_train, y_test, columns = os.Preparation()

            # Apply Gradient Boost
            grdboost = GradientBoostingClassifier(random_state=175)
            grdboost.fit(x_train, y_train)

            # Check accuracy based on train data
            y_predict1 = grdboost.predict(x_train)
            accuracy1 = accuracy_score(y_train, y_predict1)
            print("Based on train Data(grdboost): ", accuracy1)

            # Check accuracy based on test data
            y_predict2 = grdboost.predict(x_test)
            accuracy2 = accuracy_score(y_test, y_predict2)
            print("Based on test Data(grdboost): ", accuracy2, '\n')
            return accuracy2, grdboost, columns

        except Exception as e:
            return e

    # Using XGBoost Classifier Algorithm
    def XGBClassifier(self):
        try:
            os = prepare.Preparation(self.path, self.format)
            x_train, x_test, y_train, y_test, columns = os.Preparation()

            # Apply XGBoost
            xgboost1 = XGBClassifier(random_state=175, use_label_encoder=False)
            xgboost1.fit(x_train, y_train)

            # Check accuracy based on train data
            y_predict1 = xgboost1.predict(x_train)
            accuracy1 = accuracy_score(y_train, y_predict1)
            print("Based on train Data(xgboost1): ", accuracy1)

            # Check accuracy based on test data
            y_predict2 = xgboost1.predict(x_test)
            accuracy2 = accuracy_score(y_test, y_predict2)
            print("Based on test Data(xgboost1): ", accuracy2, '\n')

            # Imporove the accuracy score, so that model is less overfitted model
            # Apply GridSearch
            param_grid = {
                ' learning_rate': [1, 0.5, 0.1, 0.01, 0.001],
                'max_depth': [3, 5, 10, 20],
                'n_estimators': [10, 50, 100, 200]
            }
            grid_search = GridSearchCV(XGBClassifier(objective='binary:logistic'), param_grid, verbose=3)
            # grid_search.fit(x_train, y_train)
            # print(grid_search.best_params_)
            dct = {' learning_rate': 0.5, 'max_depth': 10, 'n_estimators': 100}
            learning_rate, max_depth, n_estimators = tuple(value for key, value in dct.items())
            xgboost = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators,
                                    random_state=175, use_label_encoder=False)
            xgboost.fit(x_train, y_train)

            # Check accuracy based on train data
            y_predict11 = xgboost.predict(x_train)
            accuracy11 = accuracy_score(y_train, y_predict11)
            print("Based on train Data(xgboost2): ", accuracy11)

            # Check accuracy based on test data
            y_predict22 = xgboost.predict(x_test)
            accuracy22 = accuracy_score(y_test, y_predict22)
            print("Based on test Data(xgboost2): ", accuracy22, '\n')
            print(columns)

            # Compare based on accuracy (accuracy2, accuracy22)
            if accuracy2 > accuracy22:
                return accuracy2, xgboost1, columns
            else:
                return accuracy22, xgboost, columns

        except Exception as e:
            return e


if __name__ == "__main__":
    path1 = "..\Data\Prepared Data\winequality-white.csv"
    path2 = "..\Data\Prepared Data\winequality-red.csv"
    f = "csv"
    pp = Model_Training(path2, f)
    pp.LogisticRegression()
    pp.DecisionTreeClassifier()
    pp.RandomForestClassifier()
    pp.GradientBoostingClassifier()
    pp.XGBClassifier()
