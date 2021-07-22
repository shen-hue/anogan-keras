"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

visualization_metrics.py

Note: Use PCA or tSNE for generated and original data visualization
"""

# Necessary packages
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def visualization(ori_data, generated_data, analysis):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
      - analysis: tsne or pca
    """
    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data,
                                        np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat,
                                            np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))

    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(pca_results[:, 0], pca_results[:, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

        ax.legend()
        plt.title('PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y_pca')
        plt.savefig('sine_CA_plot')
        # plt.show()

    elif analysis == 'tsne':

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

        ax.legend()

        plt.title('t-SNE plot')
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
        plt.savefig('result_sin/sine_t-SNE_plot')
        # plt.show()

## confusion matrix visualization
def visualization_confusion_matrix(true_negative, true_positive,false_positive,false_negative):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
      - analysis: tsne or pca
    """
    # Analysis sample size (for faster computation)
    no = list()
    no.append(len(true_negative))
    no.append(len(true_positive))
    no.append(len(false_positive))
    no.append(len(false_negative))
    idx_t_n = np.random.permutation(len(true_negative))[:]
    idx_t_p = np.random.permutation(len(true_positive))[:]
    idx_f_p = np.random.permutation(len(false_positive))[:]
    idx_f_n = np.random.permutation(len(false_negative))[:]

    # Data preprocessing
    true_negative = np.asarray(true_negative)
    true_positive = np.asarray(true_positive)
    false_positive = np.asarray(false_positive)
    false_negative = np.asarray(false_negative)

    confusion_matrix = list()
    confusion_matrix.append(true_negative[idx_t_n])
    confusion_matrix.append(true_positive[idx_t_p])
    confusion_matrix.append(false_positive[idx_f_p])
    confusion_matrix.append(false_negative[idx_f_n])

    _, seq_len, dim = true_negative.shape
    colors_name = ["blue", "red", "yellow", "green"]
    n = sum([no_example for no_example in no])
    prep_data = np.zeros((n, seq_len))
    colors = list()
    n_example = 0
    for (no_example, con_example, color) in zip(no, confusion_matrix, colors_name):
        for i in range(no_example):
            prep_data[n_example] = np.reshape(np.mean(con_example[i, :, :], 1), [1, seq_len])
            n_example += 1
            # prep_data = np.concatenate((prep_data,
            #                                 np.reshape(np.mean(con_example[i, :, :], 1), [1, seq_len])))
        # Visualization parameter
        colors = colors + [color for j in range(no_example)]


    # TSNE anlaysis
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(prep_data)
    labels = ["true negative", "true positive", "false positive", "false negative"]

    # Plotting
    f, ax = plt.subplots(1)
    i = 0
    for (no_example, label) in zip(no, labels):
        plt.scatter(tsne_results[i:i+no_example, 0], tsne_results[i:i+no_example, 1],
                c=colors[i:i+no_example], alpha=0.2, label=label)
        i += no_example

    ax.legend()

    plt.title('t-SNE plot')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    plt.savefig('result_sin/sine_t-SNE_confusion_plot')