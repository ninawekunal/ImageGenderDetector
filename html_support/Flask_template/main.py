from flask import Flask
import views

app = Flask(__name__)


# Create a rule:
app.add_url_rule('/base', 'base', views.base)
app.add_url_rule('/', 'index', views.index)
app.add_url_rule('/faceapp', 'faceapp', views.faceapp, methods=['GET', 'POST'])


# run the flask app
if __name__ == "__main__":
    app.run(debug=True)
