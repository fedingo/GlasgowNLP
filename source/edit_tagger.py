import numpy as np


COST_REPLACE = 2
COST_INSERT  = 1
COST_DELETE  = 1
COST_KEEP    = 0


def lev(matrix, i, j, compare):
    m00 = matrix[i-1, j-1]
    if not compare:
        m00 += COST_REPLACE
    else:
        m00 += COST_KEEP
    
    m10 = matrix[i, j-1] + COST_INSERT
    m01 = matrix[i-1, j] + COST_DELETE
    
    matrix[i,j] = min([m00, m10, m01])

def build_matrix (tokens1, tokens2):
    tokens1 = ["_"] + tokens1
    tokens2 = ["_"] + tokens2
    m,n = len(tokens1), len(tokens2)
    
    lev_matrix = np.zeros([m,n])
    lev_matrix[0] = list(range(n))
    lev_matrix[:,0] = list(range(m))
    
    for i in range(1,m):
        for j in range(1,n):
            lev(lev_matrix, i, j, tokens1[i] == tokens2[j])
    
    return lev_matrix

# Delete, Insert, Replace, Keep
#
def tagger(matrix, i, j):
    steps = [matrix[i, j-1], matrix[i-1, j-1], matrix[i-1, j]]
    
    if i == 0:
        return i, j-1, "I"
    elif j == 0:
        return i-1, j, "D"
    elif matrix[i-1, j-1] == min(steps):
        if matrix[i-1, j-1] == matrix[i,j]:
            return i-1, j-1, "K"
        else:
            return i-1, j-1, "R"
    elif matrix[i, j-1] == min(steps):
        return i, j-1, "I"
    elif matrix[i-1, j] == min(steps):
        return i-1, j, "D"
        

def get_tags(matrix):
    m,n = matrix.shape
    sequence = []
    m -= 1; n -= 1
    
    while not (m == 0 and n == 0):
        m,n, tag = tagger(matrix, m,n)
        sequence.append(tag)
        
    sequence.reverse()
    return sequence