import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import os

def visualise(results_dir, file_name):
    file_path = os.path.join(results_dir, file_name)
    with open(file_path, 'rb') as f:
        results = pickle.load(f)

    df = pd.DataFrame(results[0]['classification_report']).transpose()
    precision, recall, thresh1 = results[0]['pr_curve']
    tpr, fpr , thresh2 = results[0]['roc_curve']
    plt.figure(figsize=(6, 6))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.subplot(2, 2, 1)
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
    plt.plot(recall, precision)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.subplot(2, 2, 2)
    plt.plot(thresh1, precision[:-1])
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.title('Precision Curve')
    plt.subplot(2, 2, 3)
    plt.plot(thresh1, recall[:-1])
    plt.xlabel('Threshold')
    plt.ylabel('Recall')
    plt.title('Recall Curve')
    plt.subplot(2, 2, 4)
    plt.plot(tpr, fpr)
    plt.xlabel('TPR')
    plt.ylabel('FPR')
    plt.title('ROC Curve')
    plt.savefig(os.path.join(results_dir, f"{file_name[:-4]}.pdf"))
    plt.close()
    df.to_csv(os.path.join(results_dir, f"{file_name[:-4]}_report.csv"))

results_dir = "results/"
for file in os.listdir(results_dir):
    if file.endswith(".pkl"):
        visualise(results_dir, file)
    
