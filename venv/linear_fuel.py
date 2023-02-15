import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

#1.  Учитавање скупа података и приказ првих пет редова у табели
pd.set_option('display.max_columns', 13) 
pd.set_option('display.width', 170)
data = pd.read_csv('datasets/fuel_consumption.csv')
print(data.head())

#2.  Приказ концизних информација о садржају табеле и статистичких информација о свим атрибутима скупа података.
print(data.info())                    
print(data.describe())            
print(data.describe(include=[object]))

#3.  Елиминисање примерака са недостајућим вредностима атрибута или попуњавање недостајућих вредности на основу вредности атрибута осталих примерака.
data.drop('MODELYEAR', inplace=True, axis=1)
data.fillna(data.mean(numeric_only=True).round(1), inplace=True)
data = data.dropna()
print(data.describe())  
print(data.describe(include=[object]))

#4.  Графички приказ зависности континуалних атрибута коришћењем корелационе матрице
stringData = data.select_dtypes(include='object')
numData = data.select_dtypes(include='number')
sb.heatmap(numData.corr(), annot=True, square=True, fmt='.2f')
# plt.show()

#5.  Графички приказ зависности излазног атрибута од сваког улазног континуалног атрибута расејавајући тачке по Декартовом координатном систему
plt.scatter(data['ENGINESIZE'], data['CO2EMISSIONS'])
plt.xlabel('Velicina motora u litrima')
plt.ylabel('Kolicina emisije CO2-a')
# plt.show()

plt.scatter(data['CYLINDERS'], data['CO2EMISSIONS'])
plt.xlabel('Broj cilindara u motoru')
plt.ylabel('Kolicina emisije CO2-a')
# plt.show()

plt.scatter(data['FUELCONSUMPTION_CITY'], data['CO2EMISSIONS'])
plt.xlabel('Potrošnja goriva u gradu')
plt.ylabel('Kolicina emisije CO2-a')
# plt.show()

plt.scatter(data['FUELCONSUMPTION_HWY'], data['CO2EMISSIONS'])
plt.xlabel('Potrošnja goriva na otvorenom putu')
plt.ylabel('Kolicina emisije CO2-a')
# plt.show()

plt.scatter(data['FUELCONSUMPTION_COMB'], data['CO2EMISSIONS'])
plt.xlabel('Potrošnja goriva kombinovano')
plt.ylabel('Kolicina emisije CO2-a')
# plt.show()

plt.scatter(data['FUELCONSUMPTION_COMB_MPG'], data['CO2EMISSIONS'])
plt.xlabel('Potrošnja goriva u drugoj jedinici')
plt.ylabel('Kolicina emisije CO2-a')
# plt.show()

#6.  Графички приказ зависности излазног атрибута од сваког улазног категоричког атрибута користећи одговарајући тип графика
data.boxplot(column='CO2EMISSIONS', by='MAKE', figsize=(5,6))
# plt.show()

data.boxplot(column='CO2EMISSIONS', by='MODEL', figsize=(5,6))
# plt.show()

data.boxplot(column='CO2EMISSIONS', by='VEHICLECLASS', figsize=(5,6))
# plt.show()

data.boxplot(column='CO2EMISSIONS', by='TRANSMISSION', figsize=(5,6))
# plt.show()

data.boxplot(column='CO2EMISSIONS', by='FUELTYPE', figsize=(5,6))
# plt.show()

data['FUELTYPE'].hist()
# plt.show()

#7.  Одабир атрибута који учествују у тренирању модела
data_useful = data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'FUELTYPE', 'CO2EMISSIONS']]
print(data_useful.head())

data_useful_continous = data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
sb.heatmap(data_useful_continous.corr(), annot=True, square=True, fmt='.2f')
# plt.show()

data_useful_continous = data[['CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
sb.heatmap(data_useful_continous.corr(), annot=True, square=True, fmt='.2f')
# plt.show()

#8.  Извршавање додатних трансформација над одабраним атрибутима
ohe = OneHotEncoder(dtype=int, sparse = False)
fuelType = ohe.fit_transform(data_useful.FUELTYPE.to_numpy().reshape(-1, 1))
data_useful.drop(columns=['FUELTYPE'], inplace = True) # brisanje kolona po nekom uslovu
data_useful = data_useful.join(pd.DataFrame(data=fuelType, columns=ohe.get_feature_names_out(['FUELTYPE']))) # spajanje kolona
print(data_useful.head())
data_useful.drop(columns=['FUELTYPE_D'], inplace = True)
print(data_useful.head())

#9.  Формирање тренинг и тест скупова података
data_useful = data_useful.dropna()
print(data_useful.describe())
print(data_useful.head())
y = data_useful['CO2EMISSIONS']
x = data_useful.iloc[:, lambda df: [0, 1, 2, 4, 5, 6]]
print(y.head())
print(x.head())

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123, shuffle=True)

#10. Релизација и тренирање модела користећи све наведене приступе
model = LinearRegression()
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)

#11. Приказ добијених параметара модела, вредности функције грешке и прецизности модела за све реализоване приступе
df_pred = pd.DataFrame(y_predicted, columns = ['Y Predicted'], index=x_test.index)
res_df = pd.concat([x_test, y_test, df_pred], axis=1)
print(res_df.head(30))

mae = mean_absolute_error(y_true=y_test, y_pred=y_predicted)
mse = mean_squared_error(y_true=y_test, y_pred=y_predicted)
rmse = mean_squared_error(y_true=y_test, y_pred=y_predicted, squared=False)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("Model score", model.score(x_test, y_test))

#12. Приказати додатна интересантна запажања уочена током реализација модела
data['MAKE'].hist()
# plt.show()
data['VEHICLECLASS'].hist()
# plt.show()
data['TRANSMISSION'].hist()
# plt.show()

# Realizacija mog algoritma
x_column = x_train["FUELCONSUMPTION_COMB"]
y_column = y_train
x_tested = x_test["FUELCONSUMPTION_COMB"]
print(x_column.head())
print(y_column.head())
x_sum = x_column.sum()
y_sum = y_column.sum()
xy_sum = 0
x2_sum = 0
y2_sum = 0

x_list =  list(x_column)
y_list =  list(y_column)
x_real = list(x_tested)

for i in range(len(x_list)):
    xy_sum += x_list[i]*y_list[i]
    x2_sum += x_list[i]*x_list[i]
    y2_sum += y_list[i]*y_list[i]

w1 = (len(x_list)*xy_sum - x_sum*y_sum) / (len(x_list)*x2_sum - x2_sum*x2_sum)
w0 = (y_sum- w1*x_sum)/len(x_list)
print("Funkcija: h(x) =", w1, "x +", w0)

prediction_list = []
print("Prava x vrednost:", x_real[1])
for x in range(len(y_test)):
    value = w1*x_real[x]  + w0
    prediction_list.append(value)

mine_prediction = pd.DataFrame(prediction_list, columns = ['Mine Prediction'], index=y_test.index)
data_result2 = pd.concat([y_test, df_pred, mine_prediction], axis=1)
print(data_result2.head(30))

#popravi samo ovu funkciju