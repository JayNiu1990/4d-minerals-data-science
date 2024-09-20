import numpy as np
import pandas as pd
#import more_itertools as mit
import csv
import glob
import os
import operator
import functools
import struct
import csv
path = r'C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Trial Data - Copy1'
list_sorted_path_ordered_bin = []
list_sorted_path_ordered_csv = []
list_sorted_path_ordered_txt = []

for k in range(1,3,1):      
    for j in range(0,2,1):
        #for i in range(1,10,1):
        for i in range(0,10,1):
            if j==1 and i>2:
                break
            else:
                pattern1 = '/[A-Z_]*[0-9][D][' +str(j) +']['+str(i)+'][M]' + '[2][0][2][' +str(k) +']Y*.bin'
                pathall = str('%s%s'%(path,pattern1))
                sorted_path = glob.glob(pathall)
                print(k,j,i)
                #print(sorted_path)
                list_sorted_path_ordered_bin.extend(sorted_path)
                list_sorted_path_ordered_csv.extend(sorted_path)
                list_sorted_path_ordered_txt.extend(sorted_path)

list_sorted_path_ordered_csv[159:] = [w.replace('bin','csv') for w in list_sorted_path_ordered_csv[159:]]
list_sorted_path_ordered_bin[159:]

for item1,item2 in zip(list_sorted_path_ordered_bin[159:],list_sorted_path_ordered_csv[159:]):
    path1 = item1
    with open(path1,'rb') as file:
        content = file.read()
    decoded_data = struct.unpack('H'*((len(content))//2),content)
    path2 = item2
    block_num=1000
    tonnage_num=144
    grade_num=145
    with open(path2, "w",newline='') as file2:
        writer=csv.writer(file2)
        list_tonnage=[]
        list_grade=[]
        header=['TIMECOUNT','TPH','GRADE']
        list_sec = []
        for i, item in enumerate(list(decoded_data)):
            if i!=0 and i%tonnage_num==0:
                list_tonnage.append(item)
                tonnage_num+=1000
            elif i!=0 and i%grade_num==0:
                list_grade.append(item)
                grade_num+=1000
        # for i in range(int(len(decoded_data))):
        #     list_sec.append(decoded_data[13+i*1000]) 
        list_timecount=list(range(1,len(list_tonnage)))
        writer.writerow(header)
        writer.writerows(zip(list_timecount,list_tonnage,list_grade))
'''
###################unpack unsigned 16bits to base10, extract tonnage/grade to csv
filepath1 = 'C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\PLC_CMP_Data_For10D02M2021Y.bin'
with open(filepath1,'rb') as file1:
    content = file1.read()
decoded_data = struct.unpack('H'*((len(content))//2),content)
filepath2 = 'C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\PLC_CMP_Data_For10D02M2021Y_1.csv'
block_num=1000
tonnage_num=144
grade_num=145
with open(filepath2, "w",newline='') as file2:
    writer=csv.writer(file2)
    list_tonnage=[]
    list_grade=[]
    header=['TIMECOUNT','TPH','GRADE']
    for i, item in enumerate(list(decoded_data)):
        if i!=0 and i%tonnage_num==0:
            list_tonnage.append(item)
            tonnage_num+=1000
        elif i!=0 and i%grade_num==0:
            list_grade.append(item)
            grade_num+=1000
    list_timecount=list(range(1,len(list_tonnage)))
    writer.writerow(header)
    writer.writerows(zip(list_timecount,list_tonnage,list_grade))
        # elif i!=0 and i%grade_num==0: 
        #     else:
        #         pass
''' 
''' ##################write into txt
# with open(filepath2, "w") as file2:
#     for i, item in enumerate(list(decoded_data)):
#         if i!=0 and (i)%(block_num-1)==0:
#             print(i)
#             file2.write('%s'%item+',\r\n')
#         else:
#             file2.write('%s'%item+',')
            
#         elif (i-145)%(block_num-1)==0:
'''       
#writer.writerow([decoded_data[w]])

# df11 = pd.read_csv(r'C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Trial Data - Copy1\PLC_CMP_Data_For12D10M2021Y.csv')
# df22 = pd.read_csv(r'C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Trial Data - Copy\PLC_CMP_Data_For12D10M2021Y-1.csv')
# df11.equals(df22)

