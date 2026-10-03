import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from .data import CLASSES

def summarize(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if not 0 < threshold < 1:
        raise ValueError('Threshold must be between 0 and 1.')
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        'samples': len(labels), 'threshold': threshold,
        'accuracy': float(accuracy_score(labels, predictions)),
        'roc_auc': float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None,
        'pneumonia_recall': float(tp / (tp + fn)) if tp + fn else None,
        'normal_specificity': float(tn / (tn + fp)) if tn + fp else None,
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
        'classification_report': classification_report(
            labels, predictions, labels=[0, 1], target_names=CLASSES,
            output_dict=True, zero_division=0),
    }

def save_evaluation(directory, paths, labels, probabilities, threshold=0.5):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metrics = summarize(labels, probabilities, threshold)
    (directory / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    with (directory / 'predictions.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['path', 'true_label', 'pneumonia_score', 'predicted_label'])
        for path, label, probability in zip(paths, labels, probabilities):
            writer.writerow([path, CLASSES[int(label)], float(probability),
                             CLASSES[int(probability >= threshold)]])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
    matrix = np.array(metrics['confusion_matrix'])
    axes[0].imshow(matrix, cmap='Blues')
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(matrix[i, j]), ha='center', va='center')
    axes[0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASSES, yticklabels=CLASSES,
                xlabel='Predicted label', ylabel='True label', title='Confusion matrix')
    if metrics['roc_auc'] is not None:
        fpr, tpr, _ = roc_curve(labels, probabilities)
        axes[1].plot(fpr, tpr, label=f"AUC {metrics['roc_auc']:.3f}")
        axes[1].legend()
    axes[1].plot([0, 1], [0, 1], '--', color='gray')
    axes[1].set(xlabel='False positive rate', ylabel='True positive rate', title='ROC curve')
    fig.savefig(directory / 'evaluation.png', dpi=160)
    plt.close(fig)
    return metrics
