import numpy as np

def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    y = []  #best sequence path
    #scores = [row[:] for row in emission_scores] #Store emission score at their respective place in viterbi matrix
    scores = [[float("-inf") for a in range(L)] for b in range (N)]
    backpointers = [[-1 for a in range(L)] for b in range(N)]

    #Initializing for x1 (since no previous score)
    for i in xrange(L):
        scores[0][i] = start_scores[i] + emission_scores[0][i]

    #print(emission_scores)

    for i in xrange(1,N):     #word xi
        for j in xrange(L):   #tag ti
            max  = float("-inf")
            idx = -1
            for k in range(L):
                current = emission_scores[i][j] + scores[i-1][k] + trans_scores[k][j]
                if (current > max):
                    max = current
                    idx = k
            scores[i][j] = max
            backpointers[i][j] = idx

    scores[N - 1][0] = scores[N - 1][0] + end_scores[0]
    max = scores[N-1][0]
    idx = 0
    for j in xrange (1,L):
        scores[N-1][j] = scores[N-1][j] + end_scores[j]
        if (scores[N-1][j] > max):
            max = scores[N - 1][j]
            idx = j

    y.append(idx)
    last = backpointers[N-1][idx]
    for j in range(N-2,-1,-1):
        y.append(last)
        last = backpointers[j][last]


    for i in range(0,N/2):
        temp = y[i]
        y[i] = y[N-i-1]
        y[N-i-1] = temp

    #score set to 0
    return (max, y)
