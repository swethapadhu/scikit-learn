from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_gaussian_quantiles
from sklearn.metrics import mean_average_precision_score
from sklearn.metrics import plot_multiclass_precision_recall_curve


# Construct dataset (samples=2000, features=2, classes=5)
n_classes = 5
random_state = 3
X, y = make_gaussian_quantiles(n_samples=2000, n_features=2,
                                 n_classes=n_classes, random_state=random_state)

# Visualize dataset
plt.scatter(X[:,0], X[:,1], marker='o', c=y, s=25, edgecolor='k')
plt.title("Synthetic multi-class dataset")
plt.show()

# Preprocess & Create train and test dataset
y = label_binarize(y, classes=[0, 1, 2, 3, 4])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,
                                                    random_state=random_state)

# Classifier
classifier = OneVsRestClassifier(svm.SVC(random_state=random_state))
classifier.fit(X_train, y_train)
y_score = classifier.decision_function(X_test)

# Implemented mean_average_precision
mean_average_precision = mean_average_precision_score(y_test, y_score)
print(mean_average_precision)

#====================================================================================
# plot precision recall curve for multi-class
display = plot_multiclass_precision_recall_curve(classifier, X_test, y_test, name='SVM')
plt.show()