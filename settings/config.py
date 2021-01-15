import os


## App settings
name = "Fuel Efficiency Forecast"

host = "0.0.0.0"

port = int(os.environ.get("PORT", 5000))

debug = False

rcontacts = "https://www.linkedin.com/in/ennio-maldonado/"

code = "https://github.com/techno1731/RFR_Sklearn_Dash"

fontawesome = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'



## File system
root = os.path.dirname(os.path.dirname(__file__)) + "/"



## DB
