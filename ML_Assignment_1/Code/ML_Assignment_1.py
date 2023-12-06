import os
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
import seaborn as sns
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
d_direc = '/Users/rohinkumar/Desktop/ML_Assignment_1/Dataset/'

train = os.path.join(d_direc, 'a9a.txt')
test = os.path.join(d_direc, 'a9a.t')

x_train, y_train = load_svmlight_file(train)
x_test, y_test = load_svmlight_file(test)

# Check the shapes of the datasets
print(f'Training set: {x_train.shape}')
print(f'Test set: {x_test.shape}')

# Reshape the testing dataset to match the number of features in the training dataset
num_features_train = x_train.shape[1]
num_features_test = x_test.shape[1]

if num_features_test < num_features_train:
    # Add a column of zeros to x_test
    zeros_column = np.zeros((x_test.shape[0], num_features_train - num_features_test))
    x_test = np.hstack((x_test.toarray(), zeros_column))

print(f'Training set (after reshape): {x_train.shape}')
print(f'Test set (after reshape): {x_test.shape}')

# Visualizing the Sparse Matrix
plt.figure(figsize=(10, 5))
plt.spy(x_train, markersize=2, aspect='auto', markeredgecolor='red', markeredgewidth=0.5)
plt.xlabel('Features')
plt.ylabel('Samples')
plt.title('Sparse Matrix Visualization of a9a Dataset')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('sparse_x_train.png', dpi=300)
plt.show()

# Accuracy on the training set with different values of C

#C_vals = [0.01,0.05,0.1,0.5,1]

#accuracies = {}

#for i in C_vals:
 #   svm_classify = svm.SVC(kernel='linear',C=i)
  #  accuracy = cross_val_score(svm_classify, x_train, y_train, cv=3)
   # mean_accuracy = accuracy.mean()
    #accuracies[i] = mean_accuracy

#for i, accuracy in accuracies.items():
 #   print(f'C_vals={i}: Accuracy = {accuracy:.5f}')
    
    
    
# Accuracy on the training set with different values of C and gamma

#param_grid = {
 #   'gamma': [0.01, 0.05, 0.1, 0.5, 1],
  #  'C': [0.01, 0.05, 0.1, 0.5, 1]
#}

#clf_rbf = svm.SVC(kernel='rbf')
#grid_search = GridSearchCV(estimator=clf_rbf, param_grid=param_grid, cv=3, scoring='accuracy')
#grid_search.fit(x_train, y_train)
#results = grid_search.cv_results_

#for gamma in [0.01, 0.05, 0.1, 0.5, 1]:
 #   print(f'g={gamma}', end=' ')
  #  for C in [0.01, 0.05, 0.1, 0.5, 1]:
   #     index = np.where((results['param_gamma'] == gamma) & (results['param_C'] == C))[0][0]
    #    mean_test_score = results['mean_test_score'][index]
     #   print(f'C={C}: {mean_test_score:.5f}', end=' ')
    #print()
    
clf_rbf = svm.SVC(kernel='rbf', gamma=0.1, C=1)
clf_rbf.fit(x_train, y_train) 
y_pred = clf_rbf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy:", accuracy)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Print precision and recall
print("Precision:", precision)
print("Recall:", recall)
