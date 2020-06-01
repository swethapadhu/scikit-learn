from .base import _check_classifer_response_method

from .. import average_precision_score
from .. import precision_recall_curve
from .. import mean_average_precision_score

from ...utils import check_matplotlib_support
from ...utils.validation import _deprecate_positional_args
from ...base import is_classifier


class MultiClassPrecisionRecallDisplay:
    """Precision Recall visualization.

    It is recommend to use :func:`~sklearn.metrics.plot_multiclass_precision_recall_curve`
    to create a visualizer. All parameters are stored as attributes.

    Read more in the :ref:`User Guide <visualizations>`.

    Parameters
    -----------
    precision : dictionary
        Precision values.

    recall : dictionary
        Recall values.

    average_precision : dictionary, default=None
        Average precision. If None, the average precision is not shown.

    mean_average_precision : float, default=None
        mean average precision. If None, the mean average precision is not shown.

    estimator_name : str, default=None
        Name of estimator. If None, then the estimator name is not shown.

    Attributes
    ----------
    lines_ : matplotlib Artist
        Precision recall curves.

    labels_ : matplotlib Labels
        Label for precision recall curves
    ax_ : matplotlib Axes
        Axes with precision recall curves.

    figure_ : matplotlib Figure
        Figure containing the curves.
    """
    def __init__(self, precision, recall, *, mean_average_precision=None,
                 average_precision=None, estimator_name=None):
        self.precision = precision
        self.recall = recall
        self.average_precision = average_precision
        self.estimator_name = estimator_name
        self.mean_average_precision = mean_average_precision
        self.n_classes = len(precision.keys())

    def plot(self, ax=None, *, name=None, **kwargs):
        """Plot visualization.

        Extra keyword arguments will be passed to matplotlib's `plot`.

        Parameters
        ----------
        ax : Matplotlib Axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        name : str, default=None
            Name of precision recall curve for labeling. If `None`, use the
            name of the estimator.

        **kwargs : dict
            Keyword arguments to be passed to matplotlib's `plot`.

        Returns
        -------
        display : :class:`~sklearn.metrics.MultiClassPrecisionRecallDisplay`
            Object that stores computed values.
        """
        check_matplotlib_support("MultiClassPrecisionRecallDisplay.plot")
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        name = self.estimator_name if name is None else name
        self.lines_, self.labels_ = [], []

        for i in range(self.n_classes):
            l, = ax.plot(self.recall[i], self.precision[i], lw=2)
            self.lines_.append(l)
            self.labels_.append('class {0} AP = {1:0.2f}'
                          ''.format(i, self.average_precision[i]))
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set(xlabel="Recall", ylabel="Precision")
        ax.legend(self.lines_, self.labels_, prop=dict(size=14), loc='lower left')

        if self.mean_average_precision is not None and name is not None:
            ax.set_title(f"{name} (mAP = "f"{self.mean_average_precision:0.2f})")
        elif self.mean_average_precision is not None:
            ax.set_title(f"mAP = "f"{self.mean_average_precision:0.2f}")
        elif name is not None:
            plt.title(name)

        self.ax_ = ax
        self.figure_ = ax.figure
        return self


@_deprecate_positional_args
def plot_multiclass_precision_recall_curve(estimator, X, y_true, *, ax=None,
                                response_method="auto", name=None, **kwargs):
    """Plot Precision Recall Curve for multiclass classifiers.

        Extra keyword arguments will be passed to matplotlib's `plot`.

        Parameters
        ----------
        estimator : estimator instance
            Fitted classifier or a fitted :class:`~sklearn.pipeline.Pipeline`
            in which the last estimator is a classifier.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input values.

        y : array-like of shape (n_samples,n_classes)

        response_method : {'predict_proba', 'decision_function', 'auto'}, \
                          default='auto'
            Specifies whether to use :term:`predict_proba` or
            :term:`decision_function` as the target response. If set to 'auto',
            :term:`predict_proba` is tried first and if it does not exist
            :term:`decision_function` is tried next.

        name : str, default=None
            Name for labeling curve. If `None`, the name of the
            estimator is used.

        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is created.

        **kwargs : dict
            Keyword arguments to be passed to matplotlib's `plot`.

        Returns
        -------
        display : :class:`~sklearn.metrics.MultiClassPrecisionRecallDisplay`
            Object that stores computed values.
        """
    check_matplotlib_support("plot_multiclass_precision_recall_curve")

    classification_error = ("{} should be a multiclass classifier".format(
        estimator.__class__.__name__))
    if not is_classifier(estimator):
        raise ValueError(classification_error)

    prediction_method = _check_classifer_response_method(estimator,
                                                         response_method)
    y_score = prediction_method(X)

    if y_score.shape[1] <= 2:
        raise ValueError(classification_error)

    mean_average_precision = mean_average_precision_score(y_true, y_score)
    precision, recall, average_precision = dict(), dict(), dict()

    for i in range(y_score.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_score[:, i])

    viz = MultiClassPrecisionRecallDisplay(
        precision=precision, recall=recall,
        average_precision=average_precision, mean_average_precision=mean_average_precision,
        estimator_name=name
    )
    return viz.plot(ax=ax, name=name, **kwargs)