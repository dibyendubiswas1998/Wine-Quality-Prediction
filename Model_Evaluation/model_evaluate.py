import os
from Data_Preparation import prepare
from Model_Traning import trained_model
import pickle


# Evaluate the model based on accuracy
class Model_Evaluate:
    def __init__(self, path, format):
        self.path = path
        self.format = format

    def Evaluate(self):
        try:
            os = trained_model.Model_Training(self.path, self.format)
            acc1, logistic, cols1 = os.LogisticRegression()
            acc2, decision, cols2 = os.DecisionTreeClassifier()
            acc3, random, cols3 = os.RandomForestClassifier()
            acc4, graboost, cols4 = os.GradientBoostingClassifier()
            acc5, xgboost, cols5 = os.XGBClassifier()

            # Compare with all model based on accuracy
            if (acc1 > acc2) and (acc1 > acc3) and (acc1 > acc4) and (acc1 > acc5):
                print(acc1)
                print(logistic)

                # Dump the logistic model based on condition
                filename = "../Saved_Model/finalized_model.pickle"
                pickle.dump(logistic, open(filename, 'wb'))

                return acc1, logistic, cols1
            elif (acc2 > acc1) and (acc2 > acc3) and (acc2 > acc4) and (acc2 > acc5):
                print(acc2)
                print(decision)

                # Dump the decision tree model based on condition
                filename = "../Saved_Model/finalized_model.pickle"
                pickle.dump(decision, open(filename, 'wb'))

                return acc2, decision, cols2
            elif (acc3 > acc1) and (acc3 > acc2) and (acc3 > acc4) and (acc3 > acc5):
                print(acc3)
                print(random)

                # Dump the random forest model based on condition
                filename = "../Saved_Model/finalized_model.pickle"
                pickle.dump(random, open(filename, 'wb'))

                return acc3, random, cols3
            elif (acc4 > acc1) and (acc4 > acc2) and (acc4 > acc3) and (acc4 > acc5):
                print(acc4)
                print(graboost)

                # Dump the gradient boost model based on condition
                filename = "../Saved_Model/finalized_model.pickle"
                pickle.dump(graboost, open(filename, 'wb'))

                return acc4, graboost, cols4
            else:
                print(acc5)
                print(xgboost)

                # Dump the xgboost model based on condition
                filename = "../Saved_Model/finalized_model.pickle"
                pickle.dump(xgboost, open(filename, 'wb'))

                return acc5, xgboost, cols5

        except Exception as e:
            return e


if __name__ == "__main__":
    path1 = "..\Data\Prepared Data\winequality-white.csv"
    path2 = "..\Data\Prepared Data\winequality-red.csv"
    f = "csv"
    eval = Model_Evaluate(path2, f)
    eval.Evaluate()
