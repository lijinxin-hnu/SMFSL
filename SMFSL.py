from numpy import *
from function import *

class SMFSL:

    def __init__(self, ranks=[], nn_size=100, lamda_g=0,lamda_s=0,lamda_gs=[],max_iter=500,numbers=[],initialize="random",theta=0.01,outputfile=None):
        self.ranks=ranks
        self.nn_size = nn_size
        self.lamda_g=lamda_g
        self.lamda_s=lamda_s
        self.lamda_gs=lamda_gs
        self.max_iter = max_iter
        self.numbers=numbers
        self.initialize=initialize
        self.theta=theta
        self.outputfile=outputfile

    def compute_loss(self):
        total_error = 0

        for n in range(0,len(self.Rs)):
            S_type_i_index, S_type_j_index = self.Rs_indexes[n]
            S_ij=self.Ss[n]
            R_ij = self.Rs[n]
            G_i = self.Gs[S_type_i_index - 1]
            G_j = self.Gs[S_type_j_index - 1]
            R_ij_r = dot(G_i, dot(S_ij, G_j.T))
            error = linalg.norm(R_ij - R_ij_r, ord='fro') ** 2
            total_error+=error
        for i in range(0,len(self.Graphs_indexes)):
            lamda_gra = self.lamda_gs[i]
            if lamda_gra > 0:
                type_i_index=self.Graphs_indexes[i]
                Lap_i=self.Laps[i]
                G_i = self.Gs[type_i_index-1]
                total_error += trace(dot(dot(G_i.T, Lap_i), G_i)) * lamda_gra

        if self.lamda_g>0:
            for G in self.Gs:
                a=linalg.norm(G,ord="fro")**2
                total_error+=self.lamda_g*a

        if self.lamda_s>0:
            for S in self.Ss:
                a=linalg.norm(S,ord="fro")**2
                total_error+=self.lamda_s*a
        return total_error

    def S_deriv(self):
        S_deriv_matrixes=[]
        for n in range(0,len(self.Rs)):
            S_type_i_index, S_type_j_index = self.Rs_indexes[n]
            S_ij,R_ij =self.Ss[n],self.Rs[n]
            G_i = self.Gs[S_type_i_index - 1]
            G_j = self.Gs[S_type_j_index - 1]
            a1 = dot(dot(G_i.T, R_ij), G_j)
            a2 = dot(dot(dot(dot(G_i.T, G_i), S_ij), G_j.T), G_j)
            a3 = self.lamda_s * S_ij
            top=a1
            down=a2+a3
            down=down+finfo(float).eps
            S_deriv_matrix = sqrt(top / down)
            S_deriv_matrixes.append(S_deriv_matrix)

        return S_deriv_matrixes

    def G_deriv(self):
        G_deriv_matrixes=[]

        for n in range(0,len(self.Rs)):
            R_ij=self.Rs[n]
            R_type_i_index, R_type_j_index = self.Rs_indexes[n]
            G_i = self.Gs[R_type_i_index - 1]
            G_j = self.Gs[R_type_j_index - 1]
            S_ij = self.Ss[n]
            b1 = dot(dot(R_ij, G_j), S_ij.T)
            b2 = dot(G_i, dot(S_ij, dot(G_j.T, dot(G_j, S_ij.T))))
            b3 = dot(R_ij.T, dot(G_i, S_ij))
            b4 = dot(G_j, dot(S_ij.T, dot(G_i.T, dot(G_i, S_ij))))  #
            self.G_top[R_type_i_index - 1] = self.G_top[R_type_i_index - 1] + b1
            self.G_down[R_type_i_index - 1] = self.G_down[R_type_i_index - 1] +b2
            self.G_top[R_type_j_index - 1] = self.G_top[R_type_j_index - 1] + b3
            self.G_down[R_type_j_index - 1] = self.G_down[R_type_j_index - 1] + b4

        for n in range(0,len(self.Graphs)):
            Graph_i_type_index = self.Graphs_indexes[n]
            G_i = self.Gs[Graph_i_type_index - 1]

            lamda_gra_i=self.lamda_gs[n]
            if lamda_gra_i>0:
                Lap_i = self.Laps[n]
                c1 = dot(Lap_i, G_i) * lamda_gra_i
                self.G_top[Graph_i_type_index - 1] = self.G_top[Graph_i_type_index - 1] + where(c1 < 0, -c1, 0)
                self.G_down[Graph_i_type_index - 1] = self.G_down[Graph_i_type_index - 1] + where(c1 > 0, c1, 0)

        for n in range(0, len(self.Gs)):
            if self.lamda_g>0:
                G_i = self.Gs[n]
                d=self.lamda_g*G_i
                self.G_down[n]=self.G_down[n]+d
            top=self.G_top[n]
            down=self.G_down[n]
            down = down + finfo(float).eps
            G_deriv_matrix = sqrt(top / down)
            G_deriv_matrixes.append(G_deriv_matrix)

        for n in range(0, len(self.Gs)):
            self.G_top[n]=self.G_top[n]*0
            self.G_down[n]=self.G_down[n]*0

        return G_deriv_matrixes



    def initialization(self):
        Gs=[]
        Ss=[]
        G_top=[]
        G_down=[]

        if self.initialize == "random":
            for k in range(0, len(self.numbers)):
                rank = self.ranks[k]
                number = self.numbers[k]
                G=random.rand(number, rank)
                Gs.append(G)

            for index in self.Rs_indexes:
                type1_index, type2_index = index
                Ss.append(random.rand(self.ranks[type1_index - 1], self.ranks[type2_index - 1])/sqrt(self.ranks[type1_index - 1]*self.ranks[type2_index - 1]))


        if self.initialize == "random acol":
            for k in range(0,len(self.numbers)):
                rank = self.ranks[k]
                number = self.numbers[k]
                G_i = zeros((number, rank))
                length = 0
                current_length = 0
                lengths = []
                relation_matrixes = []
                for i in range(0, len(self.Rs)):
                    R_type_i_index, R_type_j_index = self.Rs_indexes[i]
                    if k + 1 == R_type_i_index:
                        R_ij = self.Rs[i]
                        lengths.append(R_ij.shape[1])
                        length += R_ij.shape[1]
                        relation_matrixes.append(R_ij)
                    elif k + 1 == R_type_j_index:
                        R_ij = self.Rs[i].T
                        lengths.append(R_ij.shape[1])
                        length += R_ij.shape[1]
                        relation_matrixes.append(R_ij)
                total_matrix = concatenate(relation_matrixes, axis=1)
                p_list = split_integer(length, rank)
                n = 0
                for m in range(0, rank):
                    p = p_list[n]
                    n += 1
                    if p > 1:
                        a = total_matrix[:, current_length:current_length + p - 1].sum(axis=1)
                    else:
                        a = total_matrix[:, current_length]
                    G_i[:, m] = a / p
                    current_length += p
                for n in range(0, G_i.shape[0]):
                    factor_vector = G_i[n, :]
                    if factor_vector.sum() == 0:
                        G_i[n, :] = random.rand(rank)
                Gs.append(G_i)

            for index in self.Rs_indexes:
                type1_index, type2_index = index
                Ss.append(random.rand(self.ranks[type1_index - 1], self.ranks[type2_index - 1]) * 2 / (
                        self.ranks[type1_index - 1] * self.ranks[type2_index - 1]))

        for n in range(0, len(self.numbers)):
            rank = self.ranks[n]
            number = self.numbers[n]
            G_top.append(zeros((number, rank)))
            G_down.append(zeros((number, rank)))

        return Gs, Ss,G_top,G_down


    def train(self,Rs,Rs_indexes,Graphs,Graphs_indexes=[],inter_pairs=None,noninter_pairs=None,pos_test=None,neg_test=None,Kfold=1):
        best_aupr=0
        self.Rs,self.Rs_indexes,self.Graphs,self.Graphs_indexes,self.Kfold = Rs,Rs_indexes,Graphs,Graphs_indexes,Kfold
        self.Laps=[]
        for i in range(0,len(Graphs)):
            Graph,lamda_gra=Graphs[i],self.lamda_gs[i]
            if lamda_gra>0:
                Lap=self.compute_laplacian_matrix(Graph, self.nn_size)
                self.Laps.append(Lap)

        self.Gs,self.Ss,self.G_top,self.G_down=self.initialization()
        last_log = self.compute_loss()

        for step in range(self.max_iter):

            S_grad_matrixes = self.S_deriv()

            for i in range(0,len(self.Ss)):
                S_grad_matrix=S_grad_matrixes[i]
                S_grad_matrix=(S_grad_matrix - 1)*self.theta+1
                self.Ss[i] = self.Ss[i]* S_grad_matrix

            G_grad_matrixes=self.G_deriv()

            for i in range(0,len(self.Gs)):
                G_grad_matrix=G_grad_matrixes[i]
                G_grad_matrix=(G_grad_matrix - 1)*self.theta+1
                self.Gs[i] = self.Gs[i]* G_grad_matrix

            curr_log = self.compute_loss()
            delta_log = abs(curr_log - last_log) / abs(last_log)
            print("iter:%s, curr_loss:%s, last_loss:%s, delta_loss:%s" % (step, curr_log, last_log, delta_log))


            if step %30==0:
                reconstruct=dot(dot(self.Gs[0],self.Ss[0].T),self.Gs[0].T)
                auc_val, aupr_val= evalution_auc(reconstruct,inter_pairs[pos_test,:],noninter_pairs[neg_test,:])
                if aupr_val>best_aupr:
                    auc_val, aupr_val, recall_val, precision_val, accuracy_val, f_measure_val,MCC_val = evalution_all(reconstruct,inter_pairs[pos_test,:],noninter_pairs[neg_test,:])
                    self.best_result=[auc_val, aupr_val, recall_val, precision_val, accuracy_val, f_measure_val,MCC_val]
                    best_aupr=aupr_val
                    
            if (abs(delta_log) < 1e-5 and step>=300) or step>=1000:
                break
            last_log = curr_log
        print("training complete")
    def build_KNN_matrix(self, S, nn_size):
        m, n = S.shape
        X = zeros((m, n))
        for i in range(m):
            ii = argsort(S[i, :])[::-1][:min(nn_size, n)]
            X[i, ii] = S[i, ii]
        return X

    def compute_laplacian_matrix(self, S, nn_size):
        if nn_size > 0:
            S1 = self.build_KNN_matrix(S, nn_size)
            x = sum(S1, axis=1)
            L = diag(x) - S1
        else:
            x = sum(S, axis=1)
            L = diag(x) - S
        return L

    def __str__(self):
        return "Model:  ranks:%s, nn_size:%s, theta:%s, lamda_g:%s, lamda_s:%s,lamda_gs:%s,max_iter:%s,initialize:%s" % (self.ranks, self.nn_size,self.theta, self.lamda_g, self.lamda_s,self.lamda_gs, self.max_iter,self.initialize)


