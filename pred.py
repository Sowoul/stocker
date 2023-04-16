import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow as tf

from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.screen import Screen
from kivymd.uix.button import MDRectangleFlatButton
from kivymd.uix.dialog import MDDialog

inp = """
MDTextField:
    hint_text: "S T O C K E R v1"
    pos_hint: {'center_x': 0.5, 'center_y': 0.5}
    icon_right: "movie-outline"
    size_hint_x: 0.4
    width:300
"""

KV1 = '''
BoxLayout:
    orientation: 'vertical'

    MDProgressBar:
        id: progress_bar
        max: app.A
        value: app.B
        size_hint_y: None
        height: dp(4)
'''


KV = '''
<ItemConfirm>
    on_release: root.set_icon(check)
    CheckboxLeftWidget:
        id: check
        group: "check"
MDFloatLayout:

<LoadingScreen>:
    name: "loading"
    canvas.before:
        Color:
            rgba: app.theme_cls.bg_normal
        Rectangle:
            size: self.size
            pos: self.pos
    MDFloatLayout:
        CircularProgressBar:
            id: progress
            size_hint: None, None
            size: dp(100), dp(100)
            pos_hint: {'center_x': 0.5, 'center_y': 0.6}
            max: 100
            value: 0
        MDLabel:
            text: "Loading..."
            halign: 'center'
            theme_text_color: 'Primary'
            pos_hint: {'center_x': 0.5, 'center_y': 0.4}

'''

colors = {
    "Teal": {
        "200": "#212121",
        "500": "#212121",
        "700": "#212121",
    },
    "Red": {
        "200": "#C25554",
        "500": "#C25554",
        "700": "#C25554",
    },
    "Light": {
        "StatusBar": "E0E0E0",
        "AppBar": "#202020",
        "Background": "#2E3032",
        "CardsDialogs": "#FFFFFF",
        "FlatButtonDown": "#CCCCCC",
    },
}


class Stocker(MDApp):
    def build(self):
        plt.clf()
        self.theme_cls.primary_palette = "Red"
        self.theme_cls.primary_hue = "500"
        self.theme_cls.primary_hue = "800"
        self.theme_cls.theme_style = "Dark"

        self.screen = Screen()
        btn = MDRectangleFlatButton(text='Search', pos_hint={
                                    'center_x': 0.5, 'center_y': 0.42}, line_width=1.2, on_release=self.grab)
        self.screen.add_widget(btn)
        # l1 = MDLabel(text='Hello world', halign='center',
        #              theme_text_color='Custom', text_color=(255/255.0, 100/255.0, 103/255.0, 1), font_style='Button')
        # screen.add_widget(l1)
        self.m = Builder.load_string(inp)
        self.screen.add_widget(self.m)
        self.dic = Builder.load_string(KV)
        self.screen.add_widget(self.dic)
        return self.screen

    def update_label(self, epoch, loss):
        self.epoch_label.text = f"Epoch: {epoch}\nLoss: {loss}"

    def close(self, obj):
        self.err.dismiss()

    def shut(self, obj):
        self.cri.dismiss()
    def die(self,obj):
        self.dia.dismiss()

    def grab(self, obj):
        plt.clf()
        dataset = []
        symbol = self.m.text
        data = yf.download(symbol, start="2015-01-01", end="2023-04-04")
        # extract the adjusted close price column
        dataframe = data[['Adj Close']]

        # reverse the order of the data
        dataset = dataframe.values
        try:
            # normalize the dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)
        except:
            first = MDRectangleFlatButton(
                text="Back", line_width=1.23, on_release=self.close)
            self.err=MDDialog(title="Error",text="Check your internet connection or spelling",buttons=[
                                    first,
                                ])
            self.err.open()
            return
        # split into train and test sets
        train_size = int(len(dataset) * 0.9)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size,
                              :], dataset[train_size:len(dataset), :]

        # convert an array of values into a dataset matrix

        def create_dataset(dataset, look_back=1):
            dataX, dataY = [], []
            for i in range(len(dataset)-look_back-1):
                a = dataset[i:(i+look_back), 0]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 0])
            return np.array(dataX), np.array(dataY)

        # reshape into X=t and Y=t+1
        look_back = 20
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)

        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
        epoch_num = 10

        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(4, input_shape=(look_back, 1)))
        model.add(Dense(8))
        model.add(Dense(16))
        model.add(Dense(32))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error',
                      optimizer='adam')

        model.fit(trainX, trainY, epochs=epoch_num, batch_size=32,
                  verbose=2)

        # make predictions
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        # Invert predictions to original scale
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([trainY])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])

        # Shift train predictions for plotting
        trainPredictPlot = np.empty_like(dataset)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(
            trainPredict)+look_back, :] = trainPredict

        # Shift test predictions for plotting
        testPredictPlot = np.empty_like(dataset)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict)+(look_back*2) +
                        1:len(dataset)-1, :] = testPredict

        # make predictions for future time steps
        forecast_input = testX[-1]
        forecast_input = np.reshape(
            forecast_input, (1, look_back, 1))  # reshape input

        # generate predictions for the next 60 time steps
        num_predictions = 10
        forecast = []
        for i in range(num_predictions):
            next_pred = model.predict(forecast_input)
            forecast.append(next_pred[0][0])
            next_pred_reshaped = np.reshape(next_pred, (1, 1, 1))
            forecast_input = np.append(
                forecast_input[:, 1:, :], next_pred_reshaped, axis=1)

        # reshape the forecast array to the correct shape
        forecast = np.reshape(forecast, (num_predictions, 1))

        # inverse the forecasted values to the original scale
        forecast = scaler.inverse_transform(forecast)

        # Plot baseline and predictions
        plt.subplot(211)
        plt.plot(scaler.inverse_transform(dataset), label='True data')
        plt.plot(trainPredictPlot, label='Training predictions')
        testPredictPlot = np.concatenate(
            (np.zeros((len(trainPredictPlot)-len(testPredictPlot), 1)), testPredictPlot))
        plt.plot(testPredictPlot, label='Test predictions')
        plt.plot(range(len(dataset), len(dataset) + num_predictions),
                 forecast, label='Future predictions')
        plt.title(self.m.text + "'s Stock Price")
        plt.legend()
        plt.subplot(212)
        plt.xlim(len(scaler.inverse_transform(dataset)) -
                 100+num_predictions, len(scaler.inverse_transform(dataset))+num_predictions+5)
        plt.plot(scaler.inverse_transform(dataset), label='True data')
        plt.plot(trainPredictPlot, label='Training predictions')
        testPredictPlot = np.concatenate(
            (np.zeros((len(trainPredictPlot)-len(testPredictPlot), 1)), testPredictPlot))
        plt.plot(testPredictPlot, label='Test predictions')
        plt.plot(range(len(dataset), len(dataset) + num_predictions),
                 forecast, label='Future predictions')
        plt.legend()
        self.dia = MDDialog(title="Results", text="The predicted values are shown in the plot.", size_hint=(
            1.0, 0.8), size=(600, 600))
        self.plot_widget = FigureCanvasKivyAgg(plt.gcf(), size_hint=(1, 0.9))
        self.dia.add_widget(self.plot_widget)
        self.dia.open()
        
Stocker().run()