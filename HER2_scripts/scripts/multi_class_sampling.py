
from my_libs import *

RANDOM_STATE = 42
     
# read in meta results dataframe
meta_results = pd.read_table("/home/inwosu/imbalanced_learn/Data/meta_results/meta_results_without_E_TABM_158.tsv", sep ='\t')
gene = meta_results['Gene']     
  
# read in gene expression data
cross_val_df = pd.read_table("/home/inwosu/imbalanced_learn/Data/expr_data/E_TABM_158.tsv", sep ='\t')      

# split matrix using only top "n" genes from meta results
X = cross_val_df[gene]
y = cross_val_df['ER_status'].map({'negative':0, 'positive':1})

gene_names = X.keys()

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = RANDOM_STATE)

pca = PCA(n_components=2)
enn = EditedNearestNeighbours()
smote = SMOTE(random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)

# model = make_pipeline(pca, enn, smote, knn)
model = make_pipeline(pca, enn, smote, NearMiss(version=2), StandardScaler(), LogisticRegression())

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(classification_report_imbalanced(y_test, model.predict(X_test))) #make_index_balanced_accuracy
print(make_index_balanced_accuracy(y_test, model.predict(X_test)))

# np.savetxt("y_array.txt", np.array(y))
# np.savetxt("X_array.txt", np.array(X))



# from run_class_imbalance import result_dir
# accuracy_file = os.path.join(result_dir, "accuracy_file_test.txt")
# accuracy_file = open(accuracy_file, "w")
# content = str(X)

# # file = open("file1.txt", "w+")
# accuracy_file.write(content)
# accuracy_file.close()
