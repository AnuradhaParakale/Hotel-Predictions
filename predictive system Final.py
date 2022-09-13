import numpy as np
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


input_data =(0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,51.0,150.0,45.0,371.0,105.3,1.0,0.0,8.0,5.0,151.0,1074.0,0.0,
             0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)

#Loading the saved model
loaded_model =pickle.load(open('C:/Users/anura/Desktop/Ene to End Projects/Job/trained_model.sav','rb'))

#changing the input data to numpy array
input_data_as_np_array = np.array(input_data)

#reshape the numpy array as we predicting for instances
input_data_reshape =input_data_as_np_array.reshape(1,-1)

prediction =loaded_model.predict(input_data_reshape)
print(prediction)

if (prediction[0] == 0 ):
    print('This Customer will not Checkin')
else:
    print('This Customer will do Checkin')




