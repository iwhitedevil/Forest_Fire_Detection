import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('fire_archive.csv')
df.dropna(inplace=True)
df['acq_date'] = pd.to_datetime(df['acq_date'])
df['month'] = df['acq_date'].dt.month
df['day'] = df['acq_date'].dt.day
df['year'] = df['acq_date'].dt.year
daynight_map = {"D": 1, "N": 0}
satellite_map = {"Terra": 1, "Aqua": 0}
df['daynight'] = df['daynight'].map(daynight_map)
df['satellite'] = df['satellite'].map(satellite_map)
types = pd.get_dummies(df['type'],dtype='int64')
df = pd.concat([df, types], axis=1)

df = df.rename(columns={0: 'type_0', 2: 'type_2', 3: 'type_3'})

bins = [0, 1, 2, 3, 4, 5]
labels = [1,2,3,4,5]

df['scan_binned'] = pd.cut(df['scan'], bins=bins, labels=labels)

df.drop(columns=['instrument', 'version', 'track','scan','type'],inplace=True)




y = df['confidence']
x = df.drop(['confidence', 'acq_date'], axis = 1)
x = StandardScaler().fit_transform(x)
x_train , x_test,y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


model = RandomForestRegressor(n_estimators=478,min_samples_split=3,min_samples_leaf=1,max_features='sqrt',max_depth=35)


#Fit
model.fit(x_train, y_train)
y_pred1 = model.predict(x_test)

print('first 5 test case :',y_test[:5])
print('first 5 test case prediction :',y_pred1[:5])

