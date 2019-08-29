
# coding: utf-8

# In[ ]:


import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


# ## Pre-process Data

# In[ ]:


train = pd.read_csv('../data/train.csv')

#Basic Cleaning
train = train.drop('InvoiceNo', axis = 1)
mask = train['Quantity'] <= 0
train[mask]['Quatity'] = 0
users_train = np.unique(train.CustomerID)

train['InvoiceDate'] = pd.to_datetime(train['InvoiceDate'])
train['Month'] = train['InvoiceDate'].apply(lambda x : int(x.date().strftime('%m')))
train['total_price'] = train['Quantity'] * train['UnitPrice']

train.head()


# In[ ]:


def dictionary(Stocks):
    prods = np.unique(Stocks, return_counts = False)

    product_dic = {}
    for n, prod in enumerate(prods):
        product_dic[prod] = n
    return product_dic

def trip_vector(transaction, prod_dic):
    trip_vec = np.zeros(len(prod_dic))
    
    for product in set(transaction):
        trip_vec[prod_dic[product]] = 1
    return trip_vec

def DataLoader(train_data, test_data, users, filter_data = True, n_trips = 7):
    n_users = len(users)
    n_prods = len(train_data['trip_vec'].iloc[0])
    
    #Preparing Output data
    y_prods = test_data.groupby(['CustomerID'])['StockCode'].apply(' '.join).reset_index()
    
    #Preparing Input data
    train_data = train_data.sort_values(by = ['CustomerID', 'InvoiceDate'])
    users_k = train_data.groupby('CustomerID').count()['StockCode'].reset_index().values
    if filter_data:
        mask = users_k[:,1] > n_trips
        users_k = users_k[mask]
    max_length = np.max(users_k[:,1])
    x, y = [], []
    for user in users_k:
        mask = train_data['CustomerID'] == user[0]
        mask_t = (y_prods['CustomerID'] == user[0])
        if sum(mask_t) != 0:
            user_trips = train_data[mask]
            y_trips = y_prods[mask_t]
            
            y_trips = y_trips['StockCode'].apply(lambda x : trip_vector(x.split(), prod_dic).astype(int))
            user_trips = user_trips['StockCode'].apply(lambda x : trip_vector(x.split(), prod_dic).astype(int))
            user_trips = np.concatenate(user_trips.values).ravel().reshape((user_trips.shape[0], n_prods))

            x_user = np.zeros((max_length-user[1], n_prods))
            x_user = np.vstack((x_user, user_trips))

            x.append(x_user)
            y.append(np.concatenate(y_trips.values).ravel())
    
    return np.array(x), np.array(y)


# In[ ]:


prod_dic = dictionary(train.StockCode)

trips = train.groupby(['CustomerID', 'InvoiceDate', 'Month'])['StockCode'].apply(' '.join).reset_index()
mask = trips['Month'] > 6
train_data = trips[~mask]; test_data = trips[mask]

print('Number of transactions in training data: ', train_data.shape[0])
print('Number of transactions in testing data: ', test_data.shape[0])

train_data['trip_vec'] = train_data['StockCode'].apply(lambda x : trip_vector(x.split(), prod_dic).astype(int))
train_data.head()

x, y = DataLoader(train_data, test_data, users_train, filter_data = False)
print('x: ', x.shape, '; y: ', y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# ## RNN Model

# In[ ]:


from keras.layers import Dense, Flatten, LSTM
from keras.models import Sequential
from keras.layers.embeddings import Embedding
import keras.backend as K

from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from keras.models import load_model


# In[ ]:


embed_dim = 128
lstm_out = 144
batch_size = 32

model = Sequential()
model.add(LSTM(units = lstm_out, dropout_U = 0.2, dropout_W = 0.2,
               return_sequences = True, input_shape = (x_train.shape[1], 3810)))
model.add(LSTM(units = 128, return_sequences = True))
model.add(LSTM(units = 256, return_sequences = True))
model.add(LSTM(units = 256))
model.add(Dense(x.shape[-1], activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# In[ ]:


checkpoints = ModelCheckpoint(filepath='../saved_models/nvacc{val_acc:.4f}_e{epoch:02d}.hdf5', 
                              verbose=1,monitor='val_acc', save_best_only=True)

model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 10, batch_size=32,
          verbose=1, callbacks=[checkpoints])


# In[ ]:


best_model = 'nvacc0.9609_e07.hdf5'
threshold = 0.5

model = load_model('../saved_models/' + best_model)
print('Best model loaded!!')

y_pred = model.predict(x_test)


# ## Filtered Data

# In[ ]:


x, y = DataLoader(train_data, test_data, users_train, filter_data = True, n_trips = 5)
print('x: ', x.shape, '; y: ', y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# In[ ]:


checkpoints = ModelCheckpoint(filepath='../saved_models/nvacc{val_acc:.4f}_e{epoch:02d}.hdf5',
                                  verbose=1,monitor='val_acc', save_best_only=True)

model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 100, batch_size=32,
          verbose=1, callbacks=[checkpoints])


# In[ ]:


train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

users_test = np.unique(test.CustomerID)

data = pd.concat([train, test], sort=False)
del train, test

prods = np.unique(data.StockCode, return_counts = True)


# In[ ]:


trips = data.groupby(['CustomerID', 'InvoiceDate'])['StockCode'].apply(', '.join).reset_index()
trips.tail()

