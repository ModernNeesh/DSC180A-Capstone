import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC



#Plot the data in a two dimensional feature space
def plot_data(X, y, colors = {0 : "purple", 1 : "gold"} , names = {0 : "Normal", 1 : "Abnormal"}):
    fig, ax = plt.subplots(figsize=(8, 6))

    custom_cmap = ListedColormap(list(colors.values()))
    
    # Plot samples by color and add legend
    scatter = ax.scatter(X[:, 0], X[:, 1], s=75, c=y, label=list(map(lambda i: names[i], y)), cmap = custom_cmap, edgecolors="k")
    ax.legend(handles=scatter.legend_elements()[0], labels=["Normal", "Abnormal"], loc="upper right", title="Classes")
    ax.set_xlabel("Principal Axis 1")
    ax.set_ylabel("Principal Axis 2")
    ax.set_title("Samples in two-dimensional feature space")
    plt.show()



#Plot the data in a two dimensional feature space and fit an SVM classifier to it
def plot_with_decision_boundary(
    kernel, X, y, ax=None, colors = {0 : "purple", 1 : "gold"}, names = {0 : "Normal", 1 : "Abnormal"}, title = None
):
    """
    kernel: Which kernel to use for the SVM classifier (rbf, linear, etc.)
    X: Embeddings
    y: Labels
    """
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

    if title is None:
        ax.set_title(f" Decision boundaries of {kernel} kernel in SVC")
    else:
        ax.set_title(title)

    if ax is None:
        plt.show()

    #Show accuracy of predictions
    pred = svm_classifier.predict(X)

    print(f"Accuracy: {(pred == y).mean()}")

    return svm_classifier

def save_embeddings_to_json(
    reduced_embeddings, labels, image_urls, a_ids,
    output_path="embedding_data/d3_data.json"
):
    
    reduced_embeddings = np.array(reduced_embeddings)
    labels = np.array(labels)

    if reduced_embeddings.shape[1] != 2:
        raise ValueError("expected reduced_embeddings to have shape (n_samples, 2)")

    if not (len(reduced_embeddings) == len(labels) == len(image_urls)):
        raise ValueError("lengths of embeddings, labels, and image_urls must match")

    d3_data = [
        {
            "x": float(x),
            "y": float(y),
            "label": int(label),
            "img": str(img),
            "a_id": str(a_id)
        }
        for (x, y), label, img, a_id in zip(reduced_embeddings, labels, image_urls, a_ids)
    ]

    with open(output_path, "w") as f:
        json.dump(d3_data, f)

    print(f"saved embeddings to {output_path}!")