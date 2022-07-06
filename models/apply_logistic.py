
import os
import sys

import pandas as pd
import numpy as np
from joblib import dump

from dateutil.relativedelta import relativedelta
import datetime

import geopandas as gpd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import ConfusionMatrixDisplay
#from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             accuracy_score, balanced_accuracy_score,
                             average_precision_score,precision_recall_curve)

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

class SplitTrainAndPredictLogistic:
    
    def __init__(self, copy_df, colonias, alcaldi=""):
        
        self.copy_df = copy_df
        self.alcaldi = alcaldi
        self.colonias = colonias
        self.path =  os.getcwd()
        self.train_test_table = pd.DataFrame({})
        
        
    def create_path_(self, file_path):
        
        return os.path.join(self.path, file_path)
        
    
    def train_test_df(self, print_info=True):

        """
        Create train and test sparse matrix.

        Return: Test and Train Sparse Matrix

        """
        if self.alcaldi=="":
            
            copy_df_local = self.copy_df.copy()
            
            
        else:
            
            ## Filter Municipality observations
            copy_df_local = self.copy_df[self.copy_df['alcaldi'] == self.alcaldi].copy()

        
        ## The test will be the last 3 months of data
        test_cut = copy_df_local['Hora'].max() - relativedelta(months=3)
    
        
        ## Create dummies
        copy_df_local = pd.get_dummies(copy_df_local, columns=['id_colonia', 'day_period', 'dia_semana', 'month'],
                                       prefix=["colonia", "day_per", "weekday", "month"], drop_first=True)

        ## Create the train DataFrame
        X_train = copy_df_local.query('Hora <= @test_cut')
        y_train = X_train['crimen']
        #ind_train = X_train['indice']
        
        X_train = X_train.drop(['Hora', 'crimen', 'indice', 'day', 'year', 'categoria_delito','Coordinates', 'alcaldi'], 
                               axis = 1)
        
        ## Crate the test DataFrame
        X_test = copy_df_local.query('Hora > @test_cut')
        y_test = X_test['crimen']
        ind_test = X_test['indice']
        X_test = X_test.drop(['Hora', 'crimen', 'indice', 'day', 'year', 'categoria_delito','Coordinates', 'alcaldi'], 
                             axis = 1)

        if print_info:
            ## Print shapes of datasets
            print("-"*100)

            print(f"|{self.alcaldi}|")

            print("\n")

            print("Total features: ", len(X_train.columns))

            print("Observations: ", len(X_train))

            print("\n")

            print("***TRAIN TABLE***")

            table_train = y_train.value_counts(normalize=True).to_frame('Relative').merge(
                y_train.value_counts().to_frame('Absolute'), left_index=True, right_index=True)

            print(table_train)

            print("\n")

            print("***TEST TABLE***")

            table_test = y_test.value_counts(normalize=True).to_frame('Relative').merge(
                y_test.value_counts().to_frame('Absolute'), left_index=True, right_index=True)

            print(table_test)

            print("-"*100)
            
        ## Crate a table with train-test-split results
        self.train_test_table = pd.DataFrame({"Alcaldia": [self.alcaldi],
                                              "Obs. Train": [table_train['Absolute'].sum()],
                                              "% Crime Train": [round(table_train.loc[1, "Relative"], 4)*100],
                                              "Obs. Test": [table_test['Absolute'].sum()],
                                              "% Crime Test": [round(table_test.loc[1, "Relative"], 4)*100]})

        
        ## Scale just the continuos variables
        scaler_cont_cols = ['past_near_crimes_500mts', 'past_crimes',
                           'TEMP', 'PRCP', 'CO', 'O3', 'PM10', 'area_km2', 'metro',
                           'metrobus', 'rtp', 'supers_minisupers', 'department_stores', 'banks',
                           'corporate_offices', 'restaurants', 'g_edu', 'no_healt_s',
                           'pop_per_km2', 'house_per_km2']

        ## int scaler
        scaler = StandardScaler()
        
        ## Separate continous variables frim the training dataset adn fit
        scaler.fit(X_train[scaler_cont_cols])
        
        ## Transform
        X_train_scale = scaler.transform(X_train[scaler_cont_cols])
        
        ## Convert to Pandas Dataframe
        X_train_scale = pd.DataFrame(X_train_scale, columns=scaler_cont_cols)
        
        ## Join scaled continuos variables and dummies
        X_train = X_train.drop(scaler_cont_cols, axis = 1)
        X_train.reset_index(inplace=True, drop=True)
        X_train = X_train_scale.join(X_train)
        
        
        ## Scale the x test dataset, but using the same mean and satnadar dev of the train dataset
        X_test_scale = scaler.transform(X_test[scaler_cont_cols])
        X_test_scale = pd.DataFrame(X_test_scale, columns=scaler_cont_cols)
        X_test_scale = X_test_scale.fillna(0)
        X_test = X_test.drop(scaler_cont_cols, axis = 1)
        X_test.reset_index(inplace=True)
        X_test = X_test_scale.join(X_test)
        
        ## Be sure columns arev sorted the same way
        X_test = X_test[X_train.columns]
        
        ## Convert Pandas DataFrame as Sparse Matrix
        save_columns = X_train.columns
        X_train = X_train.astype(pd.SparseDtype("float", 0)).sparse.to_coo()
        X_test = X_test.astype(pd.SparseDtype("float", 0)).sparse.to_coo()

        return X_train, X_test, y_train, y_test, save_columns, ind_test
    
    
    def fit_my_results(self, X_train, X_test, y_train, y_test,
                        bal_acc_curve=True):
        
        """
        Show results of Logistic Regression
        """
        
        ## Fit the model
        logistic = LogisticRegressionCV(max_iter=5000, 
                                        cv=StratifiedShuffleSplit(n_splits=5), Cs=10,
                                        scoring='f1', verbose=1).fit(X_train, y_train)
        
        ## Create a folder to save the Logistic Regression Model
        if not os.path.isdir(self.create_path_(r"logistic_models")):

            os.makedirs(self.create_path_(r"logistic_models"))
            
        ## Save the BLogistic Regression
         
        file_name = f"{(self.alcaldi).replace('.', '').replace(' ', '_').lower()}_logistic_model.joblib"
        dump(logistic, self.create_path_(r"logistic_models\\" + file_name)) 
        # To load again:
        #     logistic = load('filename.joblib')
        
        ## Estimate probabilities
        logistic_pred_prob = logistic.predict_proba(X_test)

        ## Extract probability of crimes or ones
        crime_prob = [prob[1] for prob in logistic_pred_prob]
        
        ## Create a grid of thresholds
        thresholds = [x/1000 for x in range(1, 1000)]

        ## Using crime probabilities from the Logistic Regression, create a list of arrays with 
        # the predicted class, using the list of thresholds
        y_pred_thresholds = [(np.array(crime_prob) > threshold).astype('float') 
                             for threshold in thresholds]

        ## Estimate the f1_score for each predicted threshold
        f1_score_l = [f1_score(y_test, y_pred, pos_label=1, average='binary') 
                      for y_pred in y_pred_thresholds]

        ## Find the thresholds that maximize the f1_score
        threshold = thresholds[np.argmax(f1_score_l)]

        ## Based on the threshol that maximize the f1_score, calculate the crime predictions
        predictions = (np.array(crime_prob) > threshold).astype('float')
        
        
        ## Graph f1_score curve
        if bal_acc_curve:
        
            fig = plt.figure(figsize=(8, 4))

            ax = fig.subplots(1, 1)
            
            sns.lineplot(x=thresholds, y=f1_score_l, ax=ax)

            plt.xlabel('Thresholds')
            plt.ylabel('F1 score')
            plt.suptitle('F1 score curve', fontsize=16)
            plt.title(f'{self.alcaldi}', fontsize=8)
            
            ## Save figure
            if not os.path.isdir(self.create_path_(r"figures\bal_acc_curve")):
                
                os.makedirs(self.create_path_(r"figures\bal_acc_curve"))
                
            fig_name = f"bal_acc_curve_{(self.alcaldi).replace('.', '').replace(' ', '_').lower()}.svg"

            plt.savefig(self.create_path_(r"figures\bal_acc_curve\\" + fig_name), format='svg', dpi=1200)

            plt.show()
        
        return logistic, predictions, crime_prob, threshold
    
    def show_my_results(self, y_test, crime_prob, predictions,
                        threshold=None, logistic=None, save_columns=None, 
                        print_res=True,show_dist=True,class_report=True,
                        conf_matrix_graph=True,precision_recall_curve_graph=True,
                        return_res_df=True):
        
        
        
        f1_score_ = f1_score(y_test, predictions, pos_label=1, average='binary')
        
        accuracy_score_ = accuracy_score(y_test, predictions)
        
        balanced_accuracy_score_ = balanced_accuracy_score(y_test, predictions)
        
        average_precision_score_ = average_precision_score(y_test, crime_prob)
        
        precision_score_ = precision_score(y_test, predictions, pos_label=1, average='binary')
        
        recall_score_ = recall_score(y_test, predictions, pos_label=1, average='binary')
  
        if print_res:
            
            print("Results of Logistic Regression:\n")
            
            print(f"   F1 Score: {f1_score_}")
            
            print(f"   Accuracy: {accuracy_score_}")
            
            print(f"   Balanced Accuracy: {balanced_accuracy_score_}")
            
            print(f"   Average Precision Score: {average_precision_score_}")
            
            print("\n")

        ## Print classification_report 
        if class_report:
            
            print("      ***CLASSIFICATION REPORT***\n")
            
            print(classification_report(y_test, predictions))
            
            print("***************************")
            
            
        ## Graph probability distribution
        if show_dist:

            fig = plt.figure(figsize=(8, 4))

            ax = fig.subplots(1, 1)

            sns.histplot(crime_prob)

            plt.xlabel('Crime probability')
            plt.ylabel('Count')

            plt.suptitle('Probability distribution', fontsize=16)
            plt.title(f'{self.alcaldi}', fontsize=8)
            
            ## Save figure
            if not os.path.isdir(self.create_path_(r"figures\prob_dist")):
                
                os.makedirs(self.create_path_(r"figures\prob_dist"))
                
            fig_name = f"prob_dist_{(self.alcaldi).replace('.', '').replace(' ', '_').lower()}.svg"

            plt.savefig(self.create_path_(r"figures\prob_dist\\" + fig_name), format='svg', dpi=1200)

            plt.show()
            
        ## Plor confusion matrix
        if conf_matrix_graph:
            
            cm = confusion_matrix(y_test, predictions)

            disp = ConfusionMatrixDisplay(confusion_matrix=cm)


            disp.plot()
            
            plt.suptitle('Confusion Matrix', fontsize=16)
            
            plt.title(f'{self.alcaldi}', fontsize=8)
            
            if not os.path.isdir(self.create_path_(r"figures\conf_matrix")):
                
                os.makedirs(self.create_path_(r"figures\conf_matrix"))
                
            fig_name = f"conf_matrix_{(self.alcaldi).replace('.', '').replace(' ', '_').lower()}.svg"

            plt.savefig(self.create_path_(r"figures\conf_matrix\\" + fig_name), format='svg', dpi=1200)

            plt.show()
            
            
        ## Graph ROC Curve
        if precision_recall_curve_graph:
            
            precision, recall, thresholds = precision_recall_curve(y_test, crime_prob)

            fig = plt.figure(figsize=(8, 5))

            ax = fig.subplots(1, 1)

            plt.plot(recall, precision, color="#2B667C")
            plt.ylim(0,.5)
            plt.xlim(0,.5)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.suptitle('Precision-Recall Curve', fontsize=16)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.suptitle('Precision-Recall Curve', fontsize=16)
            plt.title(f'{self.alcaldi}', fontsize=8)
            
            if not os.path.isdir(self.create_path_(r"figures\precision_recall_curve")):
                
                os.makedirs(self.create_path_(r"figures\precision_recall_curve"))
                
            fig_name = f"precision_recall_{(self.alcaldi).replace('.', '').replace(' ', '_').lower()}.svg"

            plt.savefig(self.create_path_(r"figures\precision_recall_curve\\" + fig_name), format='svg', dpi=1200)
            
            plt.show()   
     
        if return_res_df:
            
            ## Create a DataFrame to save all this results
            save_scores_df = pd.DataFrame({"Alcaldia": [self.alcaldi],
                                           "Threshold": [threshold],
                                           "F1 Score": [f1_score_],
                                           "Accuracy": [accuracy_score_],
                                           "Balanced Accuracy": [balanced_accuracy_score_],
                                           "precision_score": [precision_score_],
                                           "recall_score_": [recall_score_],
                                           "AUC": [average_precision_score_]})
    
            return save_scores_df
    
    
    def predict_crimes(self, X_test, ind_test, crime_prob, predictions, save_columns):
        
        ## Filter Municipality observations
        local_crimes_test = self.copy_df[self.copy_df['alcaldi'] == self.alcaldi].copy()

        ## Filter yos the indices observations, that were split into test DatFrame
        local_crimes_test = local_crimes_test[local_crimes_test['indice'].isin(ind_test)]

        ## Merge neighborhoods geometry and names
        local_crimes_test = local_crimes_test.merge(self.colonias[['id_colonia', 'geometry', 'colonia']],
                                          on='id_colonia',
                                          how='left')

        ## Convert to GeoPandas DataFrame
        local_crimes_test = gpd.GeoDataFrame(local_crimes_test, geometry='geometry')

        ## Use X_test to merge crime probability
        X_test = pd.DataFrame.sparse.from_spmatrix(X_test, columns= save_columns)

        X_test['indice'] = list(ind_test)
        X_test["proba_crimen"] = list(crime_prob)

        local_crimes_test = local_crimes_test.merge(X_test[['proba_crimen', 'indice']],
                                          on='indice',
                                          how='left')

        ## Create a column with the predictions
        local_crimes_test['predictions'] = predictions
        
        
        print("-"*100)
        print("-"*100)
        print("-"*100)

        print("\n")

        return local_crimes_test
    