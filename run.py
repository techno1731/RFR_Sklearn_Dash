from app.app import application
from settings import config

application.run_server(debug=config.debug,host=config.host, port=config.port)
