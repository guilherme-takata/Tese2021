import numpy as np 

def Collatz(x : int) -> list:
    seq = [x] # inicialização da lista com os termos da sequência
    while x != 1:
        if x % 2 == 0:
            x = x//2
        else:
            x = 3*x+1
        seq.append(x)
    return(seq)


def split_sequence(x: list, time_lag : int):
    '''
    Retorna dois arrays, o primeiro contendo as janelas e o segundo contendo o termo seguinte na sequência
    
    return_type : (<np.array>, <np.array>)
    '''

    X, y = list(), list()
    for i in range(len(x)):
        final_term = i + time_lag
        if final_term > len(x)-1 :
            break
        seq_x, seq_y = x[i:final_term], x[final_term]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)



