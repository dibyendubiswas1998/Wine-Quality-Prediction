# Wine-Quality-Prediction



### step:1	EDA (Exploratory Data Analysis):
  
    1. Load the data sets,
    2. Check the data types, shapes,
    3. Balanced the data sets based on output,
    4. Visualize the data distribution,
    5. Check the Relation with target variable,
    6. Handle the Outliers,
    7. Check the multicollinearity factor,
    8. Decting the outliers,
    9. Handle the outliers based on IQR Statistics,
    10. Again Check the data distribution after handle the outliers,
    11. Again Check the multicollinearity after handle the outliers,
    12. Find the Correlation with target,
    13. Write the Observations based on above stats,
    14. To save the data sets (as a preapared data).



### step:2	Data_Preparation:
  
    1.  Load the prepared data,
    2.  Spliting the data based on train_test_split,
    3.  Select the top 6 Features based on  mutual information gain,
    4.  Standarized the data (x_train and x_test),
    5.  Send to the Model_Traning packages for trained.



### step:3 Model_Traning:

    1.  Use different different algorithms,
    2.  Try to improve the accuracy score,
    3.  Try to create less overfitten model,
    4.  Send the all models, accuracy(for each model) to Model_Evaluation package for evaluate best model.



### step:4 Model_Evaluation:

    1.  Evaluate the best model based on higher accuracy,
    2.  Save these model as a finalized_model (as pickle format) to Save_Model folder.



### step: 5 Model_Deployment: 

    1.  Deploy the model, using django,
    2.  Then Predict the outcome.



