import numpy as np
import pickle
import streamlit as st


#Loading the saved model
loaded_model =pickle.load(open('C:/Users/anura/Desktop/Ene to End Projects/Job/trained_model.sav','rb'))

#creating function
def hotel_predictions(input_data):
    input_data = (0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,51.0,150.0,45.0,371.0,105.3,1.0,0.0,8.0,5.0,151.0,1074.0,0.0,
             0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)

    # changing the input data to numpy array
    input_data_as_np_array = np.array(input_data)

    # reshape the numpy array as we predicting for instances
    input_data_reshape = input_data_as_np_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshape)
    print(prediction)

    if (prediction[0] == 0):
        return 'This Customer will not Checkin'
    else:
        return 'This Customer will do Checkin'


def main():


    #giving a title
    st.title("Hotel Customer Prediction web app")

    #getting the input data from the user



    Age=st.text_input('Age of the person')
    DaysSinceCreation =st.text_input('Day Since creation')
    AverageLeadTime =st.text_input('Avg Lead_time')
    LodgingRevenue =st.text_input('Lodging Revenue')
    BookingsCanceled=st.text_input('Booking cancled')
    BookingsNoShowed=st.text_input('Booking not showed')
    PersonsNights =st.text_input('Persons Night')
    RoomNights =st.text_input('Room Nights')
    DaysSinceLastStay=st.text_input('No.of days since last day')
    DaysSinceFirstStay =st.text_input('No.of days since first day')



    #code for prediction
    hotelprediction =''

    #creating button for prediction
    if st.button('Hotel Prediction Test Result'):
        hotelprediction=hotel_predictions([Age,DaysSinceCreation,AverageLeadTime,LodgingRevenue,BookingsCanceled,BookingsNoShowed,PersonsNights,RoomNights,DaysSinceLastStay,DaysSinceFirstStay])
    st.success(hotelprediction)



if __name__=='__main__':
    main()
