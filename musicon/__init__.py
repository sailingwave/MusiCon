from flask import Flask

app = Flask(__name__)
#app.config.from_object('config')


from musicon import views
