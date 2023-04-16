# stocker  

An AI Based stock prediction software using python, complete with gui and backend


# Objective:  

The main objective of this project was to create an AI model that can accurately identify the curves of a stock, and predict the future values using a trained LSTM model.

# Details:

The optimizer used in this project is "adam", as it was found to be the best for our purpurse after testing, and gave the lowest loss function values of all.
The dataset is fetched from "yfinance" or "yahoo finance", considers the "closing" value of a particular stock over the years. The data is then split into 2 parts of 90:10, into training and test sets respectively.
The GUI was prepared using kivymd, making it possible to run the program across all devices like android, mac, and ios.

# Variables:

The major variables that affect the outcome are:

epoch_num: Number of epochs, more epochs usually lead to better grasp of the shape, and hence more accurate result, but we do have to keep overfitting in mind, for which, dropout layers can be added, as well as an EarlyStop layer.

look_back: This variable defines the past number of days that the model considers, while predicting the next day. 
So, if the variable is set to 20, then, for predicting the output of 21st of January, it will look at the data from 1-20. This makes the output faster and more accurate.

future_days: This is the number of days into the future that the model predicts. The accuracy drops exponentially with every passing day. 

#Screenshots:

Input Gui:
![image](https://user-images.githubusercontent.com/93905595/232271570-f972499b-7985-43d2-8e18-948d67dcead9.png)

Loss function:
![image](https://user-images.githubusercontent.com/93905595/232272056-f6196145-704c-4c23-b478-0bfb38550479.png)


Result:
Intel stock price:
![image](https://user-images.githubusercontent.com/93905595/232271853-f1b0b6db-fafd-4ebc-a8f5-c80346b35604.png)

Tesla stock price:
![image](https://user-images.githubusercontent.com/93905595/232272023-373f1179-25c7-4a0b-a6df-d9e52ba76aaa.png)



