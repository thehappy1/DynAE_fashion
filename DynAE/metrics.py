import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment as linear_assignment

nmi = normalized_mutual_info_score
ari = adjusted_rand_score

def cos_grad(grad1, grad2):
    grad1_list = []
    grad2_list = []
    for i in range(len(grad1)):
        grad1_list.append(grad1[i][0].flatten())
        grad2_list.append(grad2[i][0].flatten())
    grad1_vector = np.concatenate(grad1_list)
    grad2_vector = np.concatenate(grad2_list)
    return np.matmul(grad1_vector, grad2_vector) / ((np.linalg.norm(grad1_vector)) * (np.linalg.norm(grad2_vector)))

def compute_tsne(features, label, dataset):
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, perplexity=20, n_jobs=16, random_state=0, verbose=0).fit_transform(features)

        viz_df = pd.DataFrame(data=tsne[:10000])
        viz_df['Label'] = label[:10000]
        
        if dataset == 'fmnist':
            dict = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt",
                7: "Sneaker", 8: "Bag",
                9: "Ankle boot"}
            viz_df['Label'] = viz_df["Label"].map(dict)
            
        elif dataset == 'fpi':
            dict = {0: "Topz", 1: "Sunglasses", 2: "Casual Shoes", 3: "Shirts", 4: "Tops", 5: "Watches", 6: "Handbags",
                 7: "Sport Shoes", 8: "Heels", 9: "T-Shirts"}
            
        
        
        viz_df.to_csv(dataset + 'tsne.csv')
        plt.subplots(figsize=(8, 5))
        sns.scatterplot(x=0, y=1, hue=viz_df.Label.tolist(), legend='full', hue_order=sorted(viz_df['Label'].unique()),
                        palette=sns.color_palette("hls", n_colors=10),
                        alpha=.5,
                        data=viz_df)
        l = plt.legend(bbox_to_anchor=(-.1, 1.00, 1.1, .5), loc="lower left", markerfirst=True,
                       mode="expand", borderaxespad=0, ncol=10 + 1, handletextpad=0.01, prop={'size': 8})

        l.texts[0].set_text("")
        plt.ylabel("")
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig(dataset + '_tnse.png', dpi=150)
        plt.clf()
        
def best_cluster_fit(y_true, y_pred):
        y_true = y_true.astype(np.int64)
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1

        ind = linear_assignment(w.max() - w)
        best_fit = []
        for i in range(y_pred.size):
            for j in range(len(ind)):
                if ind[j][0] == y_pred[i]:
                    best_fit.append(ind[j][1])
        return best_fit, ind, w
        
def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    #assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    #from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size
