import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC



def plot_data(X, y, colors = {0 : "purple", 1 : "gold"} , names = {0 : "Normal", 1 : "Abnormal"}):
    fig, ax = plt.subplots(figsize=(8, 6))

    custom_cmap = ListedColormap(list(colors.values()))
    
    # Plot samples by color and add legend
    scatter = ax.scatter(X[:, 0], X[:, 1], s=75, c=y, label=list(map(lambda i: names[i], y)), cmap = custom_cmap, edgecolors="k")
    ax.legend(handles=scatter.legend_elements()[0], labels=["Normal", "Abnormal"], loc="upper right", title="Classes")
    ax.set_title("Samples in two-dimensional feature space")
    plt.show()



def plot_with_decision_boundary(
    kernel, X, y, ax=None, colors = {0 : "purple", 1 : "gold"}, names = {0 : "Normal", 1 : "Abnormal"}
):
    
    svm_classifier = SVC(kernel=kernel, C=1, gamma='scale')
    svm_classifier.fit(X, y)

    # Settings for plotting
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    custom_cmap = ListedColormap(list(colors.values()))

    # Plot decision boundary and margins
    common_params = {"estimator": svm_classifier, "X": X, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
        cmap = custom_cmap
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    ax.scatter(
        svm_classifier.support_vectors_[:, 0],
        svm_classifier.support_vectors_[:, 1],
        s=150,
        facecolors="none",
        edgecolors="k",
    )

    # Plot samples by color and add legend
    scatter = ax.scatter(X[:, 0], X[:, 1], s=30, c=y, label=list(map(lambda i: names[i], y)), cmap = custom_cmap, edgecolors="k")
    ax.legend(handles=scatter.legend_elements()[0], labels=["Normal", "Abnormal"], loc="upper right", title="Classes")

    ax.set_xlabel("Principal Axis 1")
    ax.set_ylabel("Principal Axis 2")
    ax.set_title(f" Decision boundaries of {kernel} kernel in SVC")

    if ax is None:
        plt.show()

    pred = svm_classifier.predict(X)

    print(f"Accuracy: {(pred == y).mean()}")

    return svm_classifier