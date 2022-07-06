# Crime Prediction in Mexico City

Using a Balanced Random Forest with data from the General Prosecutor of Justice of Mexico City, and other sources, this script is intended to predict crimes in Mexico City. 

[A map with the resulting probabilities of the model can be found here.](http://fjmadrigal.pythonanywhere.com/) (It takes a few seconds to load)

Follow this order to see a step by step how the data was transformed and the model was applied:

- [data_transformation.ipynb](https://github.com/madrigal2090/crime_prediction/blob/main/data_transformation.ipynb): This notebook contains all the code used to create the input information for the model.
- [feature_selection.ipynb](https://github.com/madrigal2090/crime_prediction/blob/main/feature_selection.ipynb): A Lasso regression was used to select the best features to run the machine learning models.
- [models](https://github.com/madrigal2090/crime_prediction/tree/main/models): This directory contains most of the models used to predict crimes. The model with the best performance is from a Balanced Random Forest that was run without using neighborhoods as a dummy variable [shortcut to the notebook](https://github.com/madrigal2090/crime_prediction/blob/main/models/modelo_RF_crimenes-col-non_dummies.ipynb). This directory also contains the *class object* with the steps to run [the Balanced Random Forest](https://github.com/madrigal2090/crime_prediction/blob/main/models/apply_brf.py) and the [Logit regression](https://github.com/madrigal2090/crime_prediction/blob/main/models/apply_logistic.py).

## Aggregated results form the best model

**Results of the best Balanced Random Forest:**

   - **F1 Score:** 7.8291%
   - **Recall:** 15.7226%
   - **Precision:** 5.2122%

![Distribution of predicted probabilities](https://github.com/madrigal2090/crime_prediction/blob/main/figures/prob_dist_.svg)

![Confusion matrix](https://github.com/madrigal2090/crime_prediction/blob/main/figures/conf_matrix_.svg)

![Precision-Recall curve](https://github.com/madrigal2090/crime_prediction/blob/main/figures/precision_recall_.svg)
