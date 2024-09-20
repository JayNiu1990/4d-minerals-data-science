############fuzzy logic example############
# import numpy as np
# import skfuzzy as fuzz
# import skfuzzy.membership as mf
# import matplotlib.pyplot as plt

# block_grade = np.arange(0.01, 0.81, 0.01)
# trucksensor_grade = np.arange(0.01, 0.81, 0.01)
# priorvariance = np.arange(0.01, 0.81, 0.01)

# risk = np.arange(0.01, 1.01, 0.01)
# input_block_grade = float(input("block grade: "))
# input_trucksensor_grade = float(input("truck sensor grade: "))
# input_priorvariance = float(input("prior variance: "))

# blockgrade_low = mf.trapmf(block_grade, [-1, -1, 0.4, 0.5])
# blockgrade_mid = mf.trapmf(block_grade, [0.4, 0.5, 0.5, 0.6]) 
# blockgrade_high = mf.trapmf(block_grade, [0.5, 0.6, 1, 1])       ##cut off is 0.6

# truckgrade_low = mf.trapmf(trucksensor_grade, [-1, -1, 0.4, 0.5])
# truckgrade_mid = mf.trapmf(trucksensor_grade, [0.4, 0.5, 0.5, 0.6]) 
# truckgrade_high = mf.trapmf(trucksensor_grade, [0.5, 0.6, 1, 1])       ##cut off is 0.6

# risk_low = mf.trapmf(risk, [0, 0, 0.2, 0.35]) 
# risk_mid = mf.trapmf(risk, [0.25, 0.5, 0.5, 0.75])       ##cut off is 0.6
# risk_high = mf.trapmf(risk, [0.65, 0.8, 1, 1])   

# fig, (ax0) = plt.subplots(nrows = 1, figsize =(10, 5))
# ax0.plot(block_grade, blockgrade_low, 'r', linewidth = 2, label = 'low')
# ax0.plot(block_grade, blockgrade_mid, 'b', linewidth = 2, label = 'middle')
# ax0.plot(block_grade, blockgrade_high, 'c', linewidth = 2, label = 'high')
# ax0.set_title('block grade')
# ax0.legend()

# fig, (ax0) = plt.subplots(nrows = 1, figsize =(10, 5))
# ax0.plot(block_grade, truckgrade_low, 'r', linewidth = 2, label = 'low')
# ax0.plot(block_grade, truckgrade_mid, 'b', linewidth = 2, label = 'middle')
# ax0.plot(block_grade, truckgrade_high, 'c', linewidth = 2, label = 'high')
# ax0.set_title('truck sensor grade')
# ax0.legend()
# fig, (ax0) = plt.subplots(nrows = 1, figsize =(10, 5))
# ax0.plot(risk, risk_low, 'r', linewidth = 2, label = 'low')
# ax0.plot(risk, risk_mid, 'b', linewidth = 2, label = 'middle')
# ax0.plot(risk, risk_high, 'c', linewidth = 2, label = 'high')
# ax0.set_title('truck acceptance risk')
# ax0.legend()


# blockgrade_low_fit = fuzz.interp_membership(block_grade, blockgrade_low, input_block_grade)
# blockgrade_mid_fit = fuzz.interp_membership(block_grade, blockgrade_mid, input_block_grade)
# blockgrade_high_fit = fuzz.interp_membership(block_grade, blockgrade_high, input_block_grade)

# truckgrade_low_fit = fuzz.interp_membership(block_grade, truckgrade_low, input_trucksensor_grade)
# truckgrade_mid_fit = fuzz.interp_membership(block_grade, truckgrade_mid, input_trucksensor_grade)
# truckgrade_high_fit = fuzz.interp_membership(block_grade, truckgrade_high, input_trucksensor_grade)

# rule1 = np.fmin(np.fmin(blockgrade_high_fit ,truckgrade_high_fit), risk_low)
# rule2 = np.fmin(np.fmin(blockgrade_high_fit ,truckgrade_mid_fit), risk_mid)
# rule3 = np.fmin(np.fmin(blockgrade_high_fit ,truckgrade_low_fit), risk_high)

# rule4 = np.fmin(np.fmin(blockgrade_mid_fit ,truckgrade_high_fit), risk_low)
# rule5 = np.fmin(np.fmin(blockgrade_mid_fit ,truckgrade_mid_fit), risk_mid)
# rule6 = np.fmin(np.fmin(blockgrade_mid_fit ,truckgrade_low_fit), risk_high)

# rule7 = np.fmin(np.fmin(blockgrade_low_fit ,truckgrade_high_fit), risk_low)
# rule8 = np.fmin(np.fmin(blockgrade_low_fit ,truckgrade_mid_fit), risk_mid)
# rule9 = np.fmin(np.fmin(blockgrade_low_fit ,truckgrade_low_fit), risk_high)

# out_risklow = np.fmax(np.fmax(rule1,rule4),rule7)
# out_riskmid = np.fmax(np.fmax(rule2,rule5),rule8)
# out_riskhigh = np.fmax(np.fmax(rule3,rule6),rule9)

# ###defuzzification
# out_risk =  np.fmax(np.fmax(out_risklow,out_riskmid),out_riskhigh)
# defuzzified = fuzz.defuzz(risk,out_risk,'centroid')
# result = fuzz.interp_membership(risk,out_risk,defuzzified)
import numpy as np
import skfuzzy as fuzz
import skfuzzy.membership as mf
import matplotlib.pyplot as plt
def fuzzylogic(input_block_grade,input_trucksensor_grade):
    block_grade = np.arange(0.01, 0.81, 0.01)
    trucksensor_grade = np.arange(0.01, 0.81, 0.01)
    
    risk = np.arange(0.01, 1.01, 0.01)
    # input_block_grade = float(input("block grade: "))
    # input_trucksensor_grade = float(input("truck sensor grade: "))
    
    blockgrade_low = mf.trapmf(block_grade, [-1, -1, 0.4, 0.5])
    blockgrade_mid = mf.trapmf(block_grade, [0.4, 0.5, 0.5, 0.6]) 
    blockgrade_high = mf.trapmf(block_grade, [0.5, 0.6, 1, 1])       ##cut off is 0.6
    
    truckgrade_low = mf.trapmf(trucksensor_grade, [-1, -1, 0.4, 0.5])
    truckgrade_mid = mf.trapmf(trucksensor_grade, [0.4, 0.5, 0.5, 0.6]) 
    truckgrade_high = mf.trapmf(trucksensor_grade, [0.5, 0.6, 1, 1])       ##cut off is 0.6
    
    risk_low = mf.trapmf(risk, [0, 0, 0.2, 0.35]) 
    risk_mid = mf.trapmf(risk, [0.25, 0.5, 0.5, 0.75])       ##cut off is 0.6
    risk_high = mf.trapmf(risk, [0.65, 0.8, 1, 1])   
    
    blockgrade_low_fit = fuzz.interp_membership(block_grade, blockgrade_low, input_block_grade)
    blockgrade_mid_fit = fuzz.interp_membership(block_grade, blockgrade_mid, input_block_grade)
    blockgrade_high_fit = fuzz.interp_membership(block_grade, blockgrade_high, input_block_grade)
    
    truckgrade_low_fit = fuzz.interp_membership(block_grade, truckgrade_low, input_trucksensor_grade)
    truckgrade_mid_fit = fuzz.interp_membership(block_grade, truckgrade_mid, input_trucksensor_grade)
    truckgrade_high_fit = fuzz.interp_membership(block_grade, truckgrade_high, input_trucksensor_grade)
    
    rule1 = np.fmin(np.fmin(blockgrade_high_fit ,truckgrade_high_fit), risk_low)
    rule2 = np.fmin(np.fmin(blockgrade_high_fit ,truckgrade_mid_fit), risk_mid)
    rule3 = np.fmin(np.fmin(blockgrade_high_fit ,truckgrade_low_fit), risk_high)
    
    rule4 = np.fmin(np.fmin(blockgrade_mid_fit ,truckgrade_high_fit), risk_low)
    rule5 = np.fmin(np.fmin(blockgrade_mid_fit ,truckgrade_mid_fit), risk_mid)
    rule6 = np.fmin(np.fmin(blockgrade_mid_fit ,truckgrade_low_fit), risk_high)
    
    rule7 = np.fmin(np.fmin(blockgrade_low_fit ,truckgrade_high_fit), risk_low)
    rule8 = np.fmin(np.fmin(blockgrade_low_fit ,truckgrade_mid_fit), risk_mid)
    rule9 = np.fmin(np.fmin(blockgrade_low_fit ,truckgrade_low_fit), risk_high)
    
    out_risklow = np.fmax(np.fmax(rule1,rule4),rule7)
    out_riskmid = np.fmax(np.fmax(rule2,rule5),rule8)
    out_riskhigh = np.fmax(np.fmax(rule3,rule6),rule9)
    
    ###defuzzification
    out_risk =  np.fmax(np.fmax(out_risklow,out_riskmid),out_riskhigh)
    defuzzified = fuzz.defuzz(risk,out_risk,'centroid')
    result = fuzz.interp_membership(risk,out_risk,defuzzified)
    return defuzzified

fuzzylogic(0.6,0.15)










