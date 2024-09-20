import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
DATE_START = '2015-01-01'
DATE_END = '2020-01-01'
dates = pd.date_range(DATE_START, DATE_END)
df = pd.DataFrame({
    'date': dates,
    'value': np.random.normal(0,1,dates.size)
})
df.set_index('date', inplace=True)
# plt.plot(df['value'])
# plt.ylabel('Value')
# plt.xlabel('Date')
# plt.title('Random Values')
# plt.show()

def random_walk(df, start_value=0, threshold=0.5,step_size=0.1, min_value=0, max_value=2):
    previous_value = start_value
    for index, row in df.iterrows():
        step_size_noise = np.random.normal(0,0.02)
        print(step_size_noise)
        if previous_value < min_value:
            previous_value = min_value
        if previous_value > max_value:
            previous_value = max_value
        probability = random.random()
        if probability >= threshold:
            df.loc[index, 'value'] = previous_value + step_size + step_size_noise
        else:
            df.loc[index, 'value'] = previous_value - step_size + step_size_noise
        previous_value = df.loc[index, 'value']
    return df
random_walk(df)
plt.plot(df['value'])
plt.ylabel('Value')
plt.xlabel('Date')
plt.title('Random Values')
plt.show()

#plt.hist(df['value'],bins=np.arange(-2,2,0.1))

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# #energy_df = pd.read_csv('C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\energydata_complete.csv')
# df = pd.read_csv(r'C:\Users\NIU004\OneDrive - CSIRO\Desktop\Mineral sorting\Kansanshi\Trial Data - Copy\all_grade_over_2000tonnage.csv')

# df = df.groupby(np.arange(len(df))//1).mean()
# # # # plt.plot(df['all grade over 2000tonnage'])
# df = df[0:1000]
# plt.hist(df,bins=np.arange(0,3,0.1))











