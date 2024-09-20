import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import more_itertools as mit
import csv
import glob
import os
path = r'C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Trial Data - Copy\tph_grade_Oct_Mar'
list_sorted_path_ordered_csv1 = []
list_sorted_path_ordered_csv2 = []
###2021 10-12
for k in range(1,2,1):      
    for j in range(1,2,1):
        #for i in range(1,10,1):
        for i in range(0,3,1):
            if j==1 and i>2:
                break
            else:
                pattern = '/[A-Z_]*[0-9][D][' +str(j) +']['+str(i)+'][M]' + '[2][0][2][' +str(k) +']Y*.csv'
                pathall = str('%s%s'%(path,pattern))
                sorted_path = glob.glob(pathall)
                print(k,j,i)
                #print(sorted_path)
                list_sorted_path_ordered_csv1.extend(sorted_path)
                list_sorted_path_ordered_csv2.extend(sorted_path)
###2022 01-present(03)
for k in range(2,3,1):      
    for j in range(0,2,1):
        #for i in range(1,10,1):
        for i in range(1,9,1):
            if j==1 and i>2:
                break
            else:
                pattern = '/[A-Z_]*[0-9][D][' +str(j) +']['+str(i)+'][M]' + '[2][0][2][' +str(k) +']Y*.csv'
                pathall = str('%s%s'%(path,pattern))
                sorted_path = glob.glob(pathall)
                print(k,j,i)
                #print(sorted_path)
                list_sorted_path_ordered_csv1.extend(sorted_path)
                list_sorted_path_ordered_csv2.extend(sorted_path)
###########tonnage#############
# list_sorted_path_ordered_csv2 = [w.replace('tph_grade_Oct_Mar','individual_truck_tonnage') for w in list_sorted_path_ordered_csv1]
# all_tonnage = []
# count=0
# for item1,item2 in zip(list_sorted_path_ordered_csv1,list_sorted_path_ordered_csv2):
#     data = pd.read_csv(item1, on_bad_lines='skip')
#     TPH = data.TPH.to_list()
#     GRADE = data.GRADE.to_list()
#     index = list(range(len(TPH)))
#     list1 = []
#     list_extracted = []
#     group_list1 =[]
#     group_list2 =[]
#     min_interval=75
#     max_interval=112
#     list1.append(index)
#     list1.append(TPH)
#     list1.append(GRADE)
#     data = np.array(list1)
#     Tonnage = data[1]
#     Grade = data[2]
#     idx = np.where(Tonnage>800)
#     list_extracted.append(idx)
#     list_extracted.append(Tonnage[Tonnage>800])
#     #print(list_extracted[0][0],list_extracted[1])
#     for group in mit.consecutive_groups(list_extracted[0][0]):
#         group_list1.append(list(group))
#     for i in range(len(group_list1)):
#         if min_interval-1<len(group_list1[i])<max_interval-1:
#             group_list2.append(group_list1[i])
#         else:
#             pass
#     filepath1 = item2
#     with open(filepath1, "w",newline='') as file1:
#         writer=csv.writer(file1)
#         #header=['Individual truck grade between 2min to 6 min ']
#         #writer.writerow(header)
#         #writer.writerow(list1[0:][1])
#         for i in range(len(group_list2)):#
#             selected_tonnage = []
#             selected_grade = []
#             #print(group_list2[0:][i])
#             for item in group_list2[0:][i]:
#                 if list1[2][item] ==0:
#                     pass
#                 else:
#                     selected_tonnage.append(list1[1][item])
#             writer.writerow(selected_tonnage)
#             all_tonnage.append(selected_tonnage)


###########grade#############        
list_sorted_path_ordered_csv3 = [w.replace('tph_grade_Oct_Mar','individual_truck_grade') for w in list_sorted_path_ordered_csv1]
all_grade = []
count=0
for item1,item2 in zip(list_sorted_path_ordered_csv1,list_sorted_path_ordered_csv3):
    data = pd.read_csv(item1, on_bad_lines='skip')
    TPH = data.TPH.to_list()
    GRADE = data.GRADE.to_list()
    index = list(range(len(TPH)))
    list1 = []
    list_extracted = []
    group_list1 =[]
    group_list2 =[]
    min_interval=75
    max_interval=112
    list1.append(index)
    list1.append(TPH)
    list1.append(GRADE)
    data = np.array(list1)
    Tonnage = data[1]
    Grade = data[2]
    idx = np.where(Tonnage>800)
    list_extracted.append(idx)
    list_extracted.append(Tonnage[Tonnage>800])
    #print(list_extracted[0][0],list_extracted[1])
    for group in mit.consecutive_groups(list_extracted[0][0]):
        group_list1.append(list(group))
    for i in range(len(group_list1)):
        if min_interval-1<len(group_list1[i])<max_interval-1:
            group_list2.append(group_list1[i])
        else:
            pass
    filepath1 = item2
    with open(filepath1, "w",newline='') as file1:
        writer=csv.writer(file1)
        header=['Individual truck grade between 2min to 6 min ']
        writer.writerow(header)
        #writer.writerow(list1[0:][1])
        for i in range(len(group_list2)):#
            selected_tonnage = []
            selected_grade = []
            #print(group_list2[0:][i])
            for item in group_list2[0:][i]:
                if list1[2][item] ==0:
                    pass
                else:
                    selected_grade.append((list1[2][item])/100)
            writer.writerow(selected_grade)
            all_grade.append(selected_grade)        

###rename
# string_list = []
# for i in list_sorted_path_ordered_csv1:
#     #print(i[-15:-13]) #day
#     #print(i[-12:-10]) #month
#     #print(i[-9:-5]) #year
#     string_list.append(i[-9:-5] + '-'+ i[-12:-10] + '-'+ i[-15:-13])

import os
path1 = os.listdir('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Trial Data - Copy\\individual_truck_grade\\')
for i in path1:
    os.rename('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Trial Data - Copy\\individual_truck_grade\\'+ i, 'C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\Trial Data - Copy\\individual_truck_grade\\'+i[-9:-5] + '-'+ i[-12:-10] + '-'+ i[-15:-13] +'.csv')

# df11 = pd.read_csv(r'C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Trial Data - Copy\tph_grade_Oct_Mar - Copy\PLC_CMP_Data_For13D10M2021Y.csv')
# df22 = pd.read_csv(r'C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Trial Data - Copy\tph_grade_Oct_Mar\2021-10-13.csv')
# df11.equals(df22)









            
# all_grade_no_zero =  []
# all_tonnage_no_zero = []
# for i in range(len(all_grade)):
#     if len(all_grade[i]) ==0:
#             pass
#     else:
#         all_grade_no_zero.append(all_grade[i])
#         all_tonnage_no_zero.append(all_tonnage[i])
# with open(r"C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Trial Data - Copy\2min_6min_time_elapse_individual_truck_tonnage_Oct21_Mar22.csv", "w",newline='') as file2:   
#     writer=csv.writer(file2)
#     header=['Individual truck tonnage between 2min to 5 min time elapse']
#     writer.writerow(header)
#     writer.writerows(all_tonnage_no_zero)         
    #print(list1[1][i])
# with open(r"C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Trial Data - Copy\2min_6min_time_elapse_individual_truck_grade_Oct21_Mar22.csv", "w",newline='') as file3:   
#     writer=csv.writer(file3)
#     header=['Individual truck grade between 2min to 5 min time elapse']
#     writer.writerow(header)
#     writer.writerows(all_grade_no_zero)      
    
    
# # Tonnage_each_truck= []
# # for i in range(len(all_tonnage_no_zero)):
# #     print((sum(all_tonnage_no_zero[0:][i])/len(all_tonnage_no_zero[0:][i]))*(len(all_tonnage_no_zero[0:][i]))/60/15)
# #     values = (sum(all_tonnage_no_zero[0:][i])/len(all_tonnage_no_zero[0:][i]))*(len(all_tonnage_no_zero[0:][i]))/60/15
# #     Tonnage_each_truck.append(values)
# # Grade_each_truck = []
# # for i in range(len(all_grade_no_zero)):
# #     #print((sum(all_grade[0:][i])/len(all_grade[0:][i])))
# #     if len(all_grade_no_zero[0:][i])==0:
# #         pass
# #     else:
# #         values = sum(all_grade_no_zero[0:][i])/len(all_grade_no_zero[0:][i])
# #         Grade_each_truck.append(values)
# # import matplotlib.pyplot as plt
# # plt.hist(Tonnage_each_truck,100)
# # plt.xlabel('Average tonnage on individual truck (ton)',fontsize=16)
# # plt.ylabel('Numbers of individual truck',fontsize=16)

# # plt.show()
# # import matplotlib.pyplot as plt
# # plt.hist(Grade_each_truck,100)
# # plt.xlabel('Average grade on individual truck (%)',fontsize=16)
# # plt.ylabel('Numbers of individual truck',fontsize=16)
# # plt.show()
# #print(group_list2)

#         #
#         #print(i)
        

#     #     
#     #     group_list.remove(group_list[i])
#     #print(i,len(group_list[i]))
#     # if len(group_list[i])<10:
#     #     pass
#     # else:
#     #     group_list.remove(group_list[i])
    


# #############################################################
#Extract all grade over 2000tonnage and average grade each day over 2000tonnage
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import more_itertools as mit
# import csv
# import glob
# import os
# path = r'C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Trial Data - Copy\tph_grade_Oct21_Mar22'
# list_sorted_path_ordered_csv1 = []
# list_sorted_path_ordered_csv2 = []
# ###2021 10-12
# for k in range(1,2,1):      
#     for j in range(1,2,1):
#         #for i in range(1,10,1):
#         for i in range(0,3,1):
#             if j==1 and i>2:
#                 break
#             else:
#                 pattern = '/[A-Z_]*[0-9][D][' +str(j) +']['+str(i)+'][M]' + '[2][0][2][' +str(k) +']Y*.csv'
#                 pathall = str('%s%s'%(path,pattern))
#                 sorted_path = glob.glob(pathall)
#                 print(k,j,i)
#                 #print(sorted_path)
#                 list_sorted_path_ordered_csv1.extend(sorted_path)
#                 list_sorted_path_ordered_csv2.extend(sorted_path)
# ###2022 01-present(03)
# for k in range(2,3,1):      
#     for j in range(0,2,1):
#         #for i in range(1,10,1):
#         for i in range(1,9,1):
#             if j==1 and i>2:
#                 break
#             else:
#                 pattern = '/[A-Z_]*[0-9][D][' +str(j) +']['+str(i)+'][M]' + '[2][0][2][' +str(k) +']Y*.csv'
#                 pathall = str('%s%s'%(path,pattern))
#                 sorted_path = glob.glob(pathall)
#                 print(k,j,i)
#                 #print(sorted_path)
#                 list_sorted_path_ordered_csv1.extend(sorted_path)
#                 list_sorted_path_ordered_csv2.extend(sorted_path)
        
            
# list_sorted_path_ordered_csv3 = [w.replace('tph_grade_Oct21_Mar22','grade_over_2000tonnage_each_row') for w in list_sorted_path_ordered_csv1]
# all_grade = []
# average_grade_per_day = []
# count=0
# list_time = []
# for item1,item2 in zip(list_sorted_path_ordered_csv1,list_sorted_path_ordered_csv3):
#     data = pd.read_csv(item1)
#     TPH = data.TPH.to_list()
#     grade = data.GRADE.to_list()
#     index = list(range(len(TPH)))
#     list1 = []
#     list_extracted = []
#     group_list1 =[]
#     group_list2 =[]
#     min_interval=1
#     list1.append(index)
#     list1.append(TPH)
#     list1.append(grade)
#     data = np.array(list1)
#     Tonnage = data[1]
#     Grade = data[2]
#     idx = np.where(Tonnage>2000)
#     list_extracted.append(idx)
#     list_extracted.append(Tonnage[Tonnage>2000])
#     #print(list_extracted[0][0],list_extracted[1])
#     for group in mit.consecutive_groups(list_extracted[0][0]):
#         group_list1.append(list(group))
#     for i in range(len(group_list1)):
#         if min_interval-1<len(group_list1[i]):
#             group_list2.append(group_list1[i])
#         else:
#             pass
#     filepath1 = item2
#     each_csv_grade = []
#     with open(filepath1, "w",newline='') as file1:
#         writer=csv.writer(file1)
#         header=['grade over 2000tonnage ']
#         writer.writerow(header)
#         #writer.writerow(list1[0:][1])
#         for i in range(len(group_list2)):#
#             selected_tonnage = []
#             selected_grade = []
#             #print(group_list2[0:][i])
#             for item in group_list2[0:][i]:
#                 if list1[2][item] ==0:
#                     pass
#                 else:
#                     selected_grade.append((list1[2][item])/100)
#                     each_csv_grade.append((list1[2][item])/100)
#             writer.writerow(selected_grade)
#             all_grade.append(selected_grade)   
#     average_grade_per_day.append(np.mean(np.array(each_csv_grade)))      
    
#     list_time.append(item1[123:-4])
#     with open(filepath1, "w",newline='') as file1:
#         writer=csv.writer(file1)
#         header=['grade over 2000tonnage ']
#         writer.writerow(header)
#         #writer.writerow(list1[0:][1])
#         for i in range(len(group_list2)):#
#             selected_tonnage = []
#             selected_grade = []
#             #print(group_list2[0:][i])
#             for item in group_list2[0:][i]:
#                 if list1[2][item] ==0:
#                     pass
#                 else:
#                     selected_grade.append((list1[2][item])/100)
#                     each_csv_grade.append((list1[2][item])/100)
#             writer.writerow(selected_grade)
#             all_grade.append(selected_grade)   
#     average_grade_per_day.append(np.mean(np.array(each_csv_grade)))        
#     list_time.append(item1[123:-4])

# all_grade_no_zero =  []
# for i in range(len(all_grade)):
#     if len(all_grade[i]) ==0:
#             pass
#     else:
#         all_grade_no_zero.append(all_grade[i])
# all_grade_no_zero_concat = [j for i in all_grade_no_zero for j in i]
# ###save all grade over 2000 tonnage
# with open(r"C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Trial Data - Copy\all_grade_over_2000tonnage.csv", "w",newline='') as file3:   
#     writer=csv.writer(file3)
#     header=['all grade over 2000tonnage']
#     writer.writerow(header)
#     writer.writerows(map(lambda x: [x],all_grade_no_zero_concat))  
    
# ###save average grade per day 
# with open(r"C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Trial Data - Copy\all_grade_per_day.csv", "w",newline='') as file3:   
#     writer=csv.writer(file3,delimiter=",")
#     writer.writerow(('average grade per day','time'))
#     writer.writerows(zip(list_time,average_grade_per_day))      
# data = pd.read_csv(r"C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Trial Data - Copy\all_grade_over_2000tonnage.csv")  
# idx = len(data) - 1 if len(data) % 3 else len(data)   
# data2 = data[:idx].groupby(data.index[:idx]//2).mean()
# data3 = data[:idx].groupby(data.index[:idx]//3).mean()
# data4 = data[:idx].groupby(data.index[:idx]//4).mean()
# data50 = data[:idx].groupby(data.index[:idx]//50).mean()
# Tonnage_each_truck= []
# for i in range(len(all_tonnage_no_zero)):
#     print((sum(all_tonnage_no_zero[0:][i])/len(all_tonnage_no_zero[0:][i]))*(len(all_tonnage_no_zero[0:][i]))/60/15)
#     values = (sum(all_tonnage_no_zero[0:][i])/len(all_tonnage_no_zero[0:][i]))*(len(all_tonnage_no_zero[0:][i]))/60/15
#     Tonnage_each_truck.append(values)
# Grade_each_truck = []
# for i in range(len(all_grade_no_zero)):
#     #print((sum(all_grade[0:][i])/len(all_grade[0:][i])))
#     if len(all_grade_no_zero[0:][i])==0:
#         pass
#     else:
#         values = sum(all_grade_no_zero[0:][i])/len(all_grade_no_zero[0:][i])
#         Grade_each_truck.append(values)
# import matplotlib.pyplot as plt
# plt.hist(Tonnage_each_truck,100)
# plt.xlabel('Average tonnage on individual truck (ton)',fontsize=16)
# plt.ylabel('Numbers of individual truck',fontsize=16)

# plt.show()
# import matplotlib.pyplot as plt
# plt.hist(Grade_each_truck,100)
# plt.xlabel('Average grade on individual truck (%)',fontsize=16)
# plt.ylabel('Numbers of individual truck',fontsize=16)
# plt.show()
# print(group_list2)

        #
        #print(i)
        

    #     
    #     group_list.remove(group_list[i])
    #print(i,len(group_list[i]))
    # if len(group_list[i])<10:
    #     pass
    # else:
    #     group_list.remove(group_list[i])
############################grade over 2000tonnage for each day################################
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import more_itertools as mit
# import csv
# import glob
# import os
# path = r'C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Trial Data - Copy\tph_grade_Oct21_Mar22'
# list_sorted_path_ordered_csv1 = []
# list_sorted_path_ordered_csv2 = []
# ###2021 10-12
# for k in range(1,2,1):      
#     for j in range(1,2,1):
#         #for i in range(1,10,1):
#         for i in range(0,3,1):
#             if j==1 and i>2:
#                 break
#             else:
#                 pattern = '/[A-Z_]*[0-9][D][' +str(j) +']['+str(i)+'][M]' + '[2][0][2][' +str(k) +']Y*.csv'
#                 pathall = str('%s%s'%(path,pattern))
#                 sorted_path = glob.glob(pathall)
#                 print(k,j,i)
#                 #print(sorted_path)
#                 list_sorted_path_ordered_csv1.extend(sorted_path)
#                 list_sorted_path_ordered_csv2.extend(sorted_path)
# ###2022 01-present(03)
# for k in range(2,3,1):      
#     for j in range(0,2,1):
#         #for i in range(1,10,1):
#         for i in range(1,9,1):
#             if j==1 and i>2:
#                 break
#             else:
#                 pattern = '/[A-Z_]*[0-9][D][' +str(j) +']['+str(i)+'][M]' + '[2][0][2][' +str(k) +']Y*.csv'
#                 pathall = str('%s%s'%(path,pattern))
#                 sorted_path = glob.glob(pathall)
#                 print(k,j,i)
#                 #print(sorted_path)
#                 list_sorted_path_ordered_csv1.extend(sorted_path)
#                 list_sorted_path_ordered_csv2.extend(sorted_path)
                
            
# list_sorted_path_ordered_csv3 = [w.replace('tph_grade_Oct21_Mar22','grade_over_2000tonnage_each_row_tonnage') for w in list_sorted_path_ordered_csv1]
# all_grade = []
# average_grade_per_day = []
# count=0
# list_time = []
# for item1,item2 in zip(list_sorted_path_ordered_csv1,list_sorted_path_ordered_csv3):
#     data = pd.read_csv(item1)
#     TPH = data.TPH.to_list()
#     grade = data.GRADE.to_list()
#     index = list(range(len(TPH)))
#     list1 = []
#     list_extracted = []
#     group_list1 =[]
#     group_list2 =[]
#     min_interval=1
#     list1.append(index)
#     list1.append(TPH)
#     list1.append(grade)
#     data = np.array(list1)
#     Tonnage = data[1]
#     Grade = data[2]
#     idx = np.where(Tonnage>2000)
#     list_extracted.append(idx)
#     list_extracted.append(Tonnage[Tonnage>2000])
#     #print(list_extracted[0][0],list_extracted[1])
#     for group in mit.consecutive_groups(list_extracted[0][0]):
#         group_list1.append(list(group))
#     for i in range(len(group_list1)):
#         if min_interval-1<len(group_list1[i]):
#             group_list2.append(group_list1[i])
#         else:
#             pass
#     filepath1 = item2
#     each_csv_grade = []
#     with open(filepath1, "w",newline='') as file1:
#         writer=csv.writer(file1)
#         header=['grade over 2000tonnage ']
#         writer.writerow(header)
#         #writer.writerow(list1[0:][1])
#         for i in range(len(group_list2)):#
#             selected_tonnage = []
#             selected_grade = []
#             #print(group_list2[0:][i])
#             for item in group_list2[0:][i]:
#                 if list1[2][item] ==0:
#                     pass
#                 else:
#                     selected_grade.append((list1[2][item])/100)
#                     each_csv_grade.append((list1[2][item])/100)
#             writer.writerow(selected_grade)
#             all_grade.append(selected_grade)   
#     average_grade_per_day.append(np.mean(np.array(each_csv_grade)))      
#     flat_group_list2 = [x for xs in group_list2 for x in xs]
#     list3 = []
#     list4 = []
#     for i in flat_group_list2:
#         list3.append((list1[2][i])/100)
#         list4.append((list1[1][i]))
#     # with open(filepath1, "w",newline='') as file1:
#     #     writer=csv.writer(file1, delimiter = '\n')
#     #     header=['grade over 2000tonnage ']
#     #     writer.writerow(header)
#     #     writer.writerow(list3)
#     with open(filepath1, "w",newline='') as file1:
#         writer=csv.writer(file1, delimiter = '\n')
#         header=['tonnage over 2000tonnage ']
#         writer.writerow(header)
#         writer.writerow(list4)
    
            