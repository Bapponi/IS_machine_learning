import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score

#1.  Учитавање скупа података и приказ првих пет редова у табели
pd.set_option('display.max_columns', 7) 
pd.set_option('display.width', None) 
data = pd.read_csv('datasets/cakes.csv')
print(data.head())

#2.  Приказ концизних информација о садржају табеле и статистичких информација о свим атрибутима скупа података.
print(data.info())                     
print(data.describe())                 
print(data.describe(include=[object])) 

#3.  Елиминисање примерака са недостајућим вредностима атрибута или попуњавање недостајућих вредности на основу вредности атрибута осталих примерака.
# imaju svi podaci

#4.  Графички приказ зависности континуалних атрибута коришћењем корелационе матрице
x = data.iloc[:, :-1] 
y = data['type']
sb.heatmap(x.corr(), annot=True, square=True, fmt='.2f')
# plt.show()

#5.  Графички приказ зависности излазног атрибута од сваког улазног континуалног атрибута расејавајући тачке по Декартовом координатном систему
plt.scatter(data['flour'], data['type'])
plt.xlabel('Kolicina brašna u gramima')
plt.ylabel('Tip kolača')
# plt.show()

plt.scatter(data['eggs'], data['type'])
plt.xlabel('Broj jaja u kolaču')
plt.ylabel('Tip kolača')
# plt.show()

sb.countplot(x='eggs', hue='type', data=data)
# plt.show()

plt.scatter(data['sugar'], data['type'])
plt.xlabel('Kolicina šećera u gramima')
plt.ylabel('Tip kolača')
# plt.show()

plt.scatter(data['milk'], data['type'])
plt.xlabel('Kolicina mleka u gramima')
plt.ylabel('Tip kolača')
# plt.show()

plt.scatter(data['butter'], data['type'])
plt.xlabel('Kolicina putera u gramima')
plt.ylabel('Tip kolača')
# plt.show()

plt.scatter(data['baking_powder'], data['type'])
plt.xlabel('Kolicina praška za pecivo u gramima')
plt.ylabel('Tip kolača')
# plt.show()

#6.  Графички приказ зависности излазног атрибута од сваког улазног категоричког атрибута користећи одговарајући тип графика


#7.  Одабир атрибута који учествују у тренирању модела
data = data


#8.  Извршавање додатних трансформација над одабраним атрибутима
result = x.apply(lambda iterator: ((iterator - iterator.min())/(iterator.max() - iterator.min())).round(2))
print(x)
print(result.head(50))

#9.  Формирање тренинг и тест скупова података
print(y)
x_train, x_test, y_train, y_test = train_test_split(result, y, train_size=0.7, random_state=123, shuffle=True) # deli dataFrame na train i test deo

#10. Релизација и тренирање модела користећи све наведене приступе
model = KNeighborsClassifier()
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)

#11. Приказ добијених параметара модела, вредности функције грешке и прецизности модела за све реализоване приступе
predicted_column = pd.Series(data=y_predicted, name='predictedType', index=x_test.index)

data_result = pd.concat([x_test, y_test, predicted_column], axis=1)
pd.set_option('display.max_columns', 8) 
print(data_result.head(30))

confusion_matrix1 = confusion_matrix(y_test, y_predicted)
print(confusion_matrix1)
# plt.matshow(confusion_matrix1)
# plt.show()

print("Model score:", model.score(x_test, y_test))
print("Precision score:", precision_score(y_test, y_predicted, pos_label='cupcake'))
print("Accuracy score:", accuracy_score(y_test, y_predicted))

#12. Приказати додатна интересантна запажања уочена током реализација модела
data['flour_category'] = pd.cut(data['flour'], bins=[0., 100, 200, 300, 400, 500, 600, 700, 800, np.inf])
sb.countplot(x='flour_category', hue='type', data=data)
# plt.show()

#13 realizacija mog algoritma
test_length = len(x_test.index)
train_length = len(x_train.index)

k = round(np.sqrt(train_length))
if (k%2) == 0:
    k = k + 1

print("Velicina konstante k:", k)

flour_list =  list(x_test["flour"])
eggs_list =  list(x_test["eggs"])
sugar_list =  list(x_test["sugar"])
milk_list =  list(x_test["milk"])
butter_list =  list(x_test["butter"])
powder_list =  list(x_test["baking_powder"])
flour_list1 =  list(x_train["flour"])
eggs_list1 =  list(x_train["eggs"])
sugar_list1 =  list(x_train["sugar"])
milk_list1 =  list(x_train["milk"])
butter_list1 =  list(x_train["butter"])
powder_list1 =  list(x_train["baking_powder"])
type_list =  list(y_train)
distance_list = []

for i in range(len(type_list)):
    if type_list[i] == 'cupcake':
        type_list[i] = 0
    else:
        type_list[i] = 1

print(type_list)

prediction_list = []

for x in range(test_length):
    for y in range(train_length):
        distance = np.sqrt(pow((flour_list1[y] - flour_list[x]), 2) + 
        pow((eggs_list1[y] - eggs_list[x]), 2) + 
        pow((sugar_list1[y] - sugar_list[x]), 2) + 
        pow((milk_list1[y] - milk_list[x]), 2) + 
        pow((butter_list1[y] - butter_list[x]), 2) + 
        pow((powder_list1[y] - powder_list[x]), 2))
        distance_list.append([distance, type_list[y]])

    sorted_list = sorted(distance_list, key=lambda x: x[0])
    distance_list.clear()
    # print(sorted_list)

    sum = 0
    for y in range(k):
        sum += sorted_list[y][1]

    sum = sum / k
    if sum >= 0.5:
        prediction = 'muffin'
    else:
        prediction = 'cupcake'
    prediction_list.append(prediction)

type_list1 =  list(y_test)
print("Lista tacna", type_list1)
print("Lista predikcija", prediction_list)


mine_prediction = pd.DataFrame(prediction_list, columns = ['Mine Prediction'], index=y_test.index)
data_result2 = pd.concat([y_test, predicted_column, mine_prediction], axis=1)
print(data_result2.head(30))

print("Precision score mine:", precision_score(y_test, mine_prediction, pos_label='cupcake'))
print("Accuracy score mine:", accuracy_score(y_test, mine_prediction))

