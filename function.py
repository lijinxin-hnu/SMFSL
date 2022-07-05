from numpy import *

from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, roc_curve, accuracy_score,auc,recall_score,f1_score,precision_score


def split_integer(m, n):
    assert n > 0
    quotient = int(m / n)
    remainder = m % n
    if remainder > 0:
        return [quotient] * (n - remainder) + [quotient + 1] * remainder
    if remainder < 0:
        return [quotient - 1] * -remainder + [quotient] * (n + remainder)
    return [quotient] * n



def load_data():
    numbers = []
    id_mappings = []
    Rs = []
    Rs_indexes = []
    Graphs = []
    Graphs_indexes = []

    with open("./datasets/List_Proteins_in_SL.txt", "r") as inf:
        gene_names = [line.rstrip() for line in inf]
        gene_id_mapping = dict(zip(gene_names, range(len(set(gene_names)))))
        gene_number = len(gene_names)
        numbers.append(gene_number)
        id_mappings.append(gene_id_mapping)

    gene_inter_pairs = []

    with open("./datasets/SL_Human_FinalCheck.txt", "r") as inf:
        for line in inf:
            id1, id2,  s = line.rstrip().split("\t")
            gene_inter_pairs.append((gene_id_mapping[id1], gene_id_mapping[id2]))
    inter_pairs = array(gene_inter_pairs, dtype=int)
    Rs.append(inter_pairs)
    Rs_indexes.append([1, 1])

    Graph_1 = loadtxt("./datasets/Gene_PPI_similarity.txt")
    Graphs.append(Graph_1)
    Graphs_indexes.append(1)

    Graph_2 = loadtxt("./datasets/Gene_GO_similarity.txt")
    Graphs.append(Graph_2)
    Graphs_indexes.append(1)

    return numbers, Rs, Rs_indexes, Graphs, Graphs_indexes, id_mappings, gene_names


def evalution_all(adj_rec, edges_pos_test, edges_neg_test):

    preds=adj_rec[edges_pos_test[:, 0], edges_pos_test[:, 1]]
    preds_neg=adj_rec[edges_neg_test[:, 0], edges_neg_test[:, 1]]

    preds_all = hstack((preds, preds_neg))
    labels_all = hstack((ones(len(preds)), zeros(len(preds_neg))))

    fpr, tpr, auc_thresholds = roc_curve(labels_all, preds_all)
    roc_score = auc(fpr, tpr)
    precisions, recalls, pr_thresholds = precision_recall_curve(labels_all, preds_all)
    aupr_score = auc(recalls, precisions)

    labels_all = labels_all.astype(float32)
    all_F_measure = zeros(len(pr_thresholds))
    for k in range(0, len(pr_thresholds)):
        if (precisions[k] + recalls[k]) > 0:
            all_F_measure[k] = 2 * precisions[k] * recalls[k] / (precisions[k] + recalls[k])
        else:
            all_F_measure[k] = 0
    max_index = all_F_measure.argmax()
    threshold = pr_thresholds[max_index]

    predicted_labels = zeros(len(labels_all))
    predicted_labels[preds_all > threshold] = 1
    f_measure = f1_score(labels_all, predicted_labels)
    accuracy = accuracy_score(labels_all, predicted_labels)
    precision = precision_score(labels_all, predicted_labels)
    recall = recall_score(labels_all, predicted_labels)
    TP=multiply(labels_all,predicted_labels).sum()
    TN=multiply((1-labels_all),(1-predicted_labels)).sum()
    FP=predicted_labels.sum()-TP
    FN=labels_all.sum()-TP
    MCC=(TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    print('TP%s, TN score:%s, FP:%s, FN:%s' % (TP,TN,FP,FN))


    return roc_score, aupr_score,recall,precision,accuracy,f_measure,MCC


def evalution_auc(adj_rec, edges_pos_test, edges_neg_test):

    preds=adj_rec[edges_pos_test[:, 0], edges_pos_test[:, 1]]
    preds_neg=adj_rec[edges_neg_test[:, 0], edges_neg_test[:, 1]]
    preds_all = hstack((preds, preds_neg))
    labels_all = hstack((ones(len(preds)), zeros(len(preds_neg))))

    fpr, tpr, auc_thresholds = roc_curve(labels_all, preds_all)
    roc_score = auc(fpr, tpr)
    precision, recall, pr_thresholds = precision_recall_curve(labels_all, preds_all)
    aupr_score = auc(recall, precision)

    return roc_score, aupr_score




def cross_divide(kfold,inter_pairs,nodes_num,seed=123):
    pos_tests=[]
    neg_tests=[]
    kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)


    x, y = triu_indices(nodes_num, k=1)
    neg_set = set(zip(x, y)) - set(zip(inter_pairs[:, 0], inter_pairs[:, 1])) - set(zip(inter_pairs[:, 1], inter_pairs[:, 0]))
    noninter_pairs = array(list(neg_set))

    pos_edge_kf = kf.split(inter_pairs)
    neg_edge_kf=kf.split(noninter_pairs)
    for train, test in pos_edge_kf:
        neg_train, neg_test = next(neg_edge_kf)
        pos_tests.append(test)
        neg_tests.append(neg_test)

    return inter_pairs,noninter_pairs,pos_tests,neg_tests

