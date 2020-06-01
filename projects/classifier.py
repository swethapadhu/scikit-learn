from itertools import cycle
import numpy as np
from sklearn.datasets import make_gaussian_quantiles
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.metrics import average_precision_score, precision_recall_curve, mean_average_precision, plot_precision_recall_curve
import matplotlib.pyplot as plt


# Construct dataset
n_classes = 5
random_state = 3
X, y = make_gaussian_quantiles(n_samples=2000, n_features=2,
                                 n_classes=n_classes, random_state=random_state)

# Visualize data
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

# scores
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
print(average_precision)
avg = []
for key, value in average_precision.items():
    avg.append(average_precision[key])
print(np.mean(np.array(avg)))

# Implemented mean_average_precision
print(mean_average_precision(y_test, y_score))

#====================================================================================
# plot precision recall curve for multi-class
colors = cycle(['navy', 'teal', 'darkorange', 'gold', 'cornflowerblue'])
lines, labels = [], []
for i, color in zip(range(n_classes), colors):
    l, = plt.plot(recall[i], precision[i], lw=2)
    lines.append(l)
    labels.append('class {0} AP = {1:0.2f}'
                  ''.format(i, average_precision[i]))
fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, prop=dict(size=14))
plt.show()