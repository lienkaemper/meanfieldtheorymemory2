def rates_with_noisy_tagging(r_E, r_P, p_E, p_P, p_FP, p_FN):
    p_TP = 1- p_FN #true positive: probability tagged given engram
    p_TN = 1 - p_FP #true negativeL probability not tagged given not engram 
    p_EgT = p_E*p_TP/(p_E*p_TP + p_P*p_FP) #probability engram given tagged 
    p_EgNT = p_E*p_FP/(p_E*p_FP + p_P*p_TN) #probability engram givne non-tagged 
    p_PgT = 1 -  p_EgT  #probability non engram given tagged 
    p_PgNT =  1 -  p_EgNT #probability non engram given non-tagged 

    r_tagged =  p_EgT*r_E +  p_PgT *r_P 
    r_non_tagged = p_EgNT*r_E + p_PgNT *r_P
    return r_tagged, r_non_tagged

def cor_with_noisy_tagging(c_EE, c_PE, c_PP, p_E, p_P, p_FP,  p_FN):
    p_TP = 1- p_FN #true positive: probability tagged given engram
    p_TN = 1 - p_FP #true negativeL probability not tagged given not engram 
    p_EgT = p_E*p_TP/(p_E*p_TP + p_P*p_FP) #probability engram given tagged 
    p_EgNT = p_E*p_FP/(p_E*p_FP + p_P*p_TN) #probability engram givne non-tagged 
    p_PgT = 1 -  p_EgT  #probability non engram given tagged 
    p_PgNT =  1 -  p_EgNT #probability non engran given non-tagged 

    c_TT = (p_EgT**2)*c_EE + (2*p_EgT*p_PgT) *c_PE  + (p_PgT**2)*c_PP
    c_NT = (p_EgNT*p_EgT)*c_EE + (p_PgNT*p_EgT + p_EgNT*p_PgT)*c_PE + (p_PgT*p_PgNT)*c_PP
    c_NN = (p_EgNT**2)*c_EE + (2*p_EgNT*p_PgNT) *c_PE  + (p_PgNT**2)*c_PP
    return c_TT, c_NT, c_NN
