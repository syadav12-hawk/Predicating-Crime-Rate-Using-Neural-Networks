


from keras import models
from keras import layers
import numpy as np
from sklearn import preprocessing
import pandas as pd
from keras.models import load_model
from sklearn.model_selection import train_test_split




url1 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data'
df1 = pd.read_csv(url1, error_bad_lines=False,header=None)

data1=df1.copy()

"""
data1.replace(to_replace = "?",value =0, inplace=True)  

labelencoder_X = LabelEncoder()
data1[3] = labelencoder_X.fit_transform(data1[3])



#Replace ? with mean 
for k in range(128):
    data1[k] = data1[k].astype(float)
    
for i in range(128):
    data1[i].replace(0, np.nan, inplace= True)

for m in range(128):
    mean_value=data1[m].mean()
    data1[m].fillna(mean_value,inplace= True)
  
x=data1.iloc[:,:127]
y=data1.iloc[:,127]

#Normalize whole data 
norm_data = preprocessing.normalize(data1)

#Spliting Train and Test Data
x_train=x.iloc[:1596,:]
y_train=y.iloc[:1596]


x_test=x.iloc[1596:,:]
y_test=y.iloc[1596:]
"""

#Remove Not Useful Data
features=data1.iloc[:,:127]
label=data1.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(features,label,test_size = 0.2)

delete_column = []
for column in x_train:
    if ((x_train[x_train[column]=="?"].shape[0]/(x_train.shape[0]))*100 > 50):
        delete_column.append(column)


       
x_train.drop(x_train.columns[delete_column],axis=1,inplace=True)
x_test = x_test[x_train.columns]


#Replace ? with NaN
x_train = x_train.replace('?',np.nan)
x_test = x_test.replace('?',np.nan)


#Replace NaN with Mean in Xtrain
for column in x_train:
    if x_train[x_train[column].isna()][column].shape[0] != 0:
        x_train[column].fillna(x_train[column].astype('float32').mean(),inplace =True)

#Replace NaN with Mean in Xtest
for column in x_test:
    if x_test[x_test[column].isna()][column].shape[0] != 0:
        x_test[column].fillna(x_train[column].astype('float32').mean(),inplace =True)

x_train = x_train.drop([3],axis=1)
x_test= x_test.drop([3],axis=1)


#Normalizing data
x_train = x_train.astype('float64')
x_test = x_test.astype('float64')
train_mean = x_train.mean(axis=0)
train_std = x_train.std(axis=0)
x_train_final = (x_train - train_mean)/train_std
x_test_final = (x_test - train_mean)/train_std


#Build Function to load models
def build_model():
    model=models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(32,activation='relu'))  

    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model
    
         
#K-Fold Training
k=4
num_val_sample=int(np.ceil(len(x_train)/4))
num_epochs=20
all_scores=[]
all_mea_histories=[]
valid=[]
for i in range(k):
    print("Processing fold :{}".format(i))    
    #Prepare validation fold
    x_val=x_train[i*num_val_sample:(i+1)*num_val_sample]
    y_val=y_train[i*num_val_sample:(i+1)*num_val_sample]
    
   
    a=x_train[:i*num_val_sample]
    b=x_train[(i+1)*num_val_sample:]
    part_x_train=np.concatenate((a,b),axis=0)
    
    c=y_train[:i*num_val_sample]
    d=y_train[(i+1)*num_val_sample:]
    part_y_train=np.concatenate((c,d),axis=0)
    

    #Train Using Training Fold
    model_cd=build_model()
    history=model_cd.fit(part_x_train,part_y_train,epochs=num_epochs
                  ,batch_size=1,validation_data=(x_val,y_val))
    
    model_cd.save("model_crime_data.hdf5")
                  
    mae_history=history.history['mae']
    valid.append(history.history['val_mae'])
    all_mea_histories.append(mae_history)
    
    #Record Training Performance
    #val_mse,val_mae=model_cd.evaluate(x_val,y_val,verbose=0)
    #print(f"Print MAE for Val for fold : {i} is {val_mae}")
    #all_scores.append(val_mae)


#Compute AVg MAE
avg_mea_history=[np.mean([x[i] for x in all_mea_histories]) for i in range(num_epochs)]    
avg_val_mea_history=[np.mean([y[j] for y in valid]) for j in range(num_epochs)] 


#PLot
import matplotlib.pyplot as plt
plt.plot(range(1,len(avg_val_mea_history)+1),avg_val_mea_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae Validation Error")
plt.show()


model_test=load_model("model_crime_data.hdf5")
model_final=build_model()
model_final.fit(x_train,y_train,epochs=20,batch_size=1)
test_mse,test_mae=model_final.evaluate(x_test,y_test)


print("Test MAE Score is :")
print(test_mae)




