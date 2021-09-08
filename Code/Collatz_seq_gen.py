import numpy as np 

def Collatz(x : int) -> np.array:
    seq = [x] # inicialização da lista com os termos da sequência
    while x != 1:
        if x % 2 == 0:
            x = x//2
        else:
            x = 3*x+1
        seq.append(x)
    return(seq)


def split_sequences(sequence : list, window_lag: int) -> np.array:
    '''
    Retorna um array contendo as janelas de cada ponto da série
    '''
    X, y = list(), list()
    for i in range(len(sequence)):
        final_ix = i + window_lag
        if final_ix > len(sequence)-1: # checamos se a janela está fora da nossa série temporal
            break
        subseq_x, subseq_y = sequence[i:final_ix], sequence[final_ix]# Subsequencias formadas pela janela daquele ponto
        X.append(subseq_x)
        y.append(subseq_y)
    return(np.array(X), np.array(y))



