from function import *
import numpy as np
import time
from SMFSL import *



def train():
    seed=123
    kfold=5
    gra,nn,lamda_s,lamda_g,rank=-10,4,-8,-10,160
    gra1,gra2=-10,-10
    initialize, theta, maxiter = "random acol", 2.0 ** (-3), 1000
    numbers, Rs, Rs_indexes, Graphs, Graphs_indexes, id_mappings,genenames = load_data()
    inter_pairs = Rs[0]
    inter_pairs, noninter_pairs, pos_tests, neg_tests=cross_divide(kfold, inter_pairs, numbers[0], seed=seed)
    print("Data initlized!")
    outputfile = open("datasets/results.txt", "a")

    lamda_gs=[2**gra1,2**gra2]
    auc_pair, aupr_pair, recall_pair, precision_pair, accuracy_pair, f_measure_pair, MCC_pair = [], [], [], [], [], [], []
    t = time.time()
    pos_x, pos_y = inter_pairs[:, 0], inter_pairs[:, 1]
    outputfile.write("kfold: " + str(kfold) + "\n")


    for fold in range(kfold):
        pos_test=pos_tests[fold]
        neg_test=neg_tests[fold]

        pos_test_x,pos_test_y=inter_pairs[pos_test, 0],inter_pairs[pos_test, 1]
        model = SMFSL(ranks=[160], nn_size=16, lamda_g=2**lamda_g, lamda_s=2**lamda_g,lamda_gs=lamda_gs, numbers=numbers, initialize=initialize, theta=theta,max_iter=maxiter, outputfile=outputfile)

        print(str(model))
        outputfile.write(str(model) + "\n")
        SL_Mat = np.zeros((numbers[0], numbers[0]))
        SL_Mat[pos_x, pos_y] = 1
        SL_Mat[pos_y, pos_x] = 1
        SL_Mat[pos_test_x,pos_test_y]=0
        SL_Mat[pos_test_y, pos_test_x] = 0
        Rs[0] = SL_Mat
        model.train(Rs, Rs_indexes, Graphs, Graphs_indexes=Graphs_indexes, inter_pairs=inter_pairs,noninter_pairs=noninter_pairs, pos_test=pos_test, neg_test=neg_test, Kfold=fold)
        [auc_val, aupr_val, recall_val, precision_val, accuracy_val, f_measure_val,MCC_val] = model.best_result
        auc_pair.append(auc_val)
        aupr_pair.append(aupr_val)
        recall_pair.append(recall_val)
        precision_pair.append(precision_val)
        accuracy_pair.append(accuracy_val)
        f_measure_pair.append(f_measure_val)
        MCC_pair.append(MCC_val)
        print("metrics over protein pairs: auc %.6f, aupr %.6f,recall %.6f,precision %.6f,accuracy %.6f,f_measure %.6f,MCC %.6f, time: %f\n" % (auc_val, aupr_val, recall_val, precision_val, accuracy_val, f_measure_val, MCC_val, time.time() - t))
        outputfile.write("metrics over protein pairs: auc %.6f, aupr %.6f,recall %.6f,precision %.6f,accuracy %.6f,f_measure %.6f,MCC %.6f, time: %f\n" % (auc_val, aupr_val, recall_val, precision_val, accuracy_val, f_measure_val, MCC_val, time.time() - t))
        outputfile.flush()

    m1=sum(auc_pair)/len(auc_pair)
    m2=sum(aupr_pair) / len(aupr_pair)
    m3=sum(recall_pair) / len(recall_pair)
    m4=sum(precision_pair) / len(precision_pair)
    m5=sum(accuracy_pair) / len(accuracy_pair)
    m6=sum(f_measure_pair) / len(f_measure_pair)
    m7=sum(MCC_pair) / len(MCC_pair)
    print("Average metrics over pairs: auc_mean:%.6f, aupr_mean:%.6f,recall_mean:%.6f,precision_mean:%.6f,accuracy_mean:%.6f,f_measure_mean:%.6f,MCC_mean:%.6f \n" % (m1, m2, m3, m4, m5, m6, m7))
    outputfile.write("Average metrics over pairs: auc_mean:%.6f, aupr_mean:%.6f,recall_mean:%.6f,precision_mean:%.6f,accuracy_mean:%.6f,f_measure_mean:%.6f,MCC_mean:%.6f \n" % (m1, m2, m3, m4, m5, m6, m7))
    outputfile.write("%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n\n" % (m1, m2, m3, m4, m5, m6, m7))


if __name__ == "__main__":
    train()




