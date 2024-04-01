# Import necessary libraries
import os, sys
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE
from imblearn.under_sampling import RandomUnderSampler, AllKNN, EditedNearestNeighbours, RepeatedEditedNearestNeighbours
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from imblearn.ensemble import EasyEnsembleClassifier, RUSBoostClassifier, BalancedBaggingClassifier, BalancedRandomForestClassifier


# define directories
meta_results_dir = "/home/inwosu/imbalanced_learn/Data/meta_results/"
cross_val_data_dir = "/home/inwosu/imbalanced_learn/Data/expr_data/"

result_dir = "/home/inwosu/imbalanced_learn/Data/results"

accuracy_file = os.path.join(result_dir, "accuracy_file.txt")
accuracy_file = open(accuracy_file, "w")

# at least 10 with deep- learning (Tensorflow/ Keras)
# Define resampling strategies
resampling_strategies = {
    # oversamplers
    'ROS': RandomOverSampler(sampling_strategy='auto', random_state = 42),
    'SMOTE': SMOTE(random_state = 0), # Synthetic Minority Oversampling Technique (SMOTE)   
    # 'ADA': ADASYN(sampling_strategy='minority', random_state = 0), # Adaptive Synthetic (ADASYN)  sampling method
   
    # undersampler
    'RUS': RandomUnderSampler(sampling_strategy='auto', random_state=42),
    'ENN': EditedNearestNeighbours(),
    'RENN': RepeatedEditedNearestNeighbours(),
    'ALL': AllKNN(allow_minority=True),

    # Combination samplers
    'SMOTENN': SMOTEENN(random_state=42),

    #Ensemble Methods
    'EEC' : EasyEnsembleClassifier(random_state=42),
    'RBC' : RUSBoostClassifier(random_state=42),
    'BBC' : BalancedBaggingClassifier(random_state=42),
    'BRFC' : BalancedRandomForestClassifier(random_state=42)    
}

ensemble_methods = ['EEC', 'RBC', 'BBC', 'BRFC']

# Define models
# 5 variants
#  set defaults (sampling strategy and replacement)
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42, solver='lbfgs', max_iter=3000)
    # KNN, SVM
}

for filename in os.listdir(meta_results_dir):
    # get file name        
    dataset_id = filename.replace("meta_results_without_", "")
    meta_path_name = os.path.join(meta_results_dir, "meta_results_without_" + dataset_id)
    
    # read in meta results dataframe
    meta_results = pd.read_table(meta_path_name, sep ='\t') 
    gene = meta_results['Gene']     
             
    # read in gene expression data
    cross_val_path_name = os.path.join(cross_val_data_dir, dataset_id)
    cross_val_df = pd.read_table(cross_val_path_name, sep ='\t')      

    # split matrix using only top "n" genes from meta results
    X = cross_val_df[gene]
    y = cross_val_df['ER_status'].map({'negative':0, 'positive':1})

    gene_names = X.keys()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Compare models with different resampling strategies
    for resampler_name, resampler in resampling_strategies.items():
        
        print(f"\nResampling Strategy: {resampler_name}")

        for model_name, model in models.items():

            if resampler_name in ensemble_methods:
                resampler.fit(X_train, y_train)
                y_pred = resampler.predict(X_test)
                # resampler.score(X_test, y_test)
            
            else:
                # Create pipeline with resampling strategy and model
                pipeline = Pipeline([('resampler', resampler), ('model', model)])

                # Fit the pipeline on the training data
                pipeline.fit(X_train, y_train)         
                
                # Make predictions on the test set
                y_pred = pipeline.predict(X_test)
            
            # Evaluate the model
            roc = roc_auc_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Print the results
            print(f"\nModel: {model}")
            print(f"ROC: {roc}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

            # accuracy_file.writelines(f"\ndataset: {dataset_id} \tModel: {model_name} \tAccuracy: {accuracy:.4f} \tPrecision: {precision:.4f} \tRecall: {recall:.4f} \tF1 Score: {f1:.4f}")
            accuracy_file.writelines(f"\n{dataset_id} \t{resampler_name} \t{model_name} \t{roc} \t{accuracy:.4f} \t{precision:.4f} \t{recall:.4f} \t{f1:.4f}")

accuracy_file.close()