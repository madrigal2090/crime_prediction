# Crime Prediction in Mexico City

Using data from the General Prosecutor of Justice of Mexico City, and other sources, this script is intended to predict crimes in Mexico City, using a Balanced Random Forest. 

[A map with the resulting probabilities of the model can be found her.](http://fjmadrigal.pythonanywhere.com/)

Follow this order to see a step by step how the data was transformed and the model was applied:

- [data_transformation.ipynb](https://github.com/madrigal2090/crime_prediction/blob/main/data_transformation.ipynb): This notebook contains all the code used to create the input information for the model.
- [modelo_RF_crimenes.ipynb](https://github.com/madrigal2090/crime_prediction/blob/main/modelo_RF_crimenes.ipynb): In this notebook the model was run, using also a [class object model](https://github.com/madrigal2090/crime_prediction/blob/main/apply_brf.py) that contains the steps to run the Balanced Random Forest.

## Aggregated results

**Results of Balanced Random Forest:**

   - **F1 Score:** 0.07478332162098851
   - **Accuracy:** 0.9525956997377596
   - **Balanced Accuracy:** 0.5769196984068939
   - **AUC:** 0.7448168205401235

![Probability Distribution](https://github.com/madrigal2090/crime_prediction/blob/main/figures/prob_dist/prob_dist_.svg)

![Confusion matrix](https://github.com/madrigal2090/crime_prediction/blob/main/figures/conf_matrix/conf_matrix_.svg)

![ROC Curve](https://github.com/madrigal2090/crime_prediction/blob/main/figures/roc_curve/roc_curve_.svg)


