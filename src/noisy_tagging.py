def rates_with_noisy_tagging(r_E, r_N, p_E=.5, p_N=.5, p_FP=0.0, p_FN=0.0):
    p_TP = 1- p_FN #true positive: probability tagged given engram
    p_TN = 1 - p_FP #true negative: probability not tagged given not engram 
    p_EgT = p_E*p_TP/(p_E*p_TP + p_N*p_FP) #probability engram given tagged 
    p_EgNT = p_E*p_FN/(p_E*p_FN + p_N*p_TN) #probability engram givne non-tagged 
    p_NgT = 1 -  p_EgT  #probability non engram given tagged 
    p_NgNT =  1 -  p_EgNT #probability non engram given non-tagged 

    r_tagged =  p_EgT*r_E +  p_NgT *r_N 
    r_non_tagged = p_EgNT*r_E + p_NgNT *r_N
    return r_tagged, r_non_tagged

def cor_with_noisy_tagging(c_EE, c_PE, c_PP, p_E=.5, p_N=.5, p_FP=.0,  p_FN=.0):
    p_TP = 1- p_FN #true positive: probability tagged given engram
    p_TN = 1 - p_FP #true negativeL probability not tagged given not engram 
    p_EgT = p_E*p_TP/(p_E*p_TP + p_N*p_FP) #probability engram given tagged 
    p_EgNT = p_E*p_FP/(p_E*p_FP + p_N*p_TN) #probability engram givne non-tagged 
    p_NgT = 1 -  p_EgT  #probability non engram given tagged 
    p_NgNT =  1 -  p_EgNT #probability non engran given non-tagged 

    c_TT = (p_EgT**2)*c_EE + (2*p_EgT*p_NgT) *c_PE  + (p_NgT**2)*c_PP
    c_NT = (p_EgNT*p_EgT)*c_EE + (p_NgNT*p_EgT + p_EgNT*p_NgT)*c_PE + (p_NgT*p_NgNT)*c_PP
    c_NN = (p_EgNT**2)*c_EE + (2*p_EgNT*p_NgNT) *c_PE  + (p_NgNT**2)*c_PP
    return c_TT, c_NT, c_NN
