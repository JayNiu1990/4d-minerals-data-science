import operator
import functools
import struct
import numpy as np
import csv
############# unpack bin,save to txt
filepath1 = 'C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Trial Data - Copy\\PLC_CMP_Data_For10D11M2021Y.bin'
with open(filepath1,'rb') as file1:
    content = file1.read()
decoded_data = struct.unpack('H'*((len(content))//2),content)

time1 = decoded_data[0:1000]
time2 = decoded_data[1000:2000]
time3 = decoded_data[24000:25000]



data = []
for i in range(len(decoded_data)//1000):
        data.append(decoded_data[13+i*1000]) 


idx = []
for i,item in enumerate(data):
    if item<4:
        idx.append(i)
idx.append(len(data))
        
data1 = []
for i in range(len(idx)):
    data1.append(data[idx[i]:idx[i+1]])
    
data2= []
for i,item in enumerate(data1):
    if i==0:
        data2.append(item)
    else:
        maxvalue = max(data2[i-1])
        maxvalue_list = np.zeros(len(item)) + maxvalue
        newvalue = maxvalue_list + item
        data2.append(newvalue)


    

        
        
        

    





i=1000
print(decoded_data[0+i:1000+i])#grade93

with open('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\PLC_CMP_Data_For16D01M2021Y.txt','w') as file:
    for i, item in enumerate(list(decoded_data[1000:2000])): 
        if (i+1)%20==0:
            file.write(str(item)+'\r\n')
        else:
            file.write(str(item)+' ')


# filepath1 = 'C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Trial Data\\PLC_CMP_Data_For01D03M2021Y.bin'
# with open(filepath1,'rb') as file1:
#     content = file1.read(300)
# decoded_data = struct.unpack('H'*((len(content))//2),content)
# print(decoded_data)
# print(decoded_data[144],decoded_data[145])
# with open('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\PLC_CMP_Data_For01D03M2021Y.txt','w') as file:
#     for i in decoded_data:
#         file.write(str(i)+' ')
filepath2 = 'C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\PLC_CMP_Data_For16D01M2021Y.csv'
block_num=1000
tonnage_num=144
grade_num=145

with open(filepath2, "w",newline='') as file2:
    writer=csv.writer(file2)
    list_timecount=[]
    list_tonnage=[]
    list_grade=[]
    header=['timecount','tonnage','grade']
    for i, item in enumerate(list(decoded_data)):
        if i!=0 and i%tonnage_num==0:
            list_tonnage.append(item)
            tonnage_num+=1000
        elif i!=0 and i%grade_num==0:
            list_grade.append(item)
            grade_num+=1000
    writer.writerows(zip(list_tonnage,list_grade))
    
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

        

    