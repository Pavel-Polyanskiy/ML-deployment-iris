 # Import the python file containing the ML model
from flask import Flask, request, render_template # Import flask libraries
from PIL import Image
import base64
import io
import model_new



# Initialize the flask class and specify the templates directory
app = Flask(__name__,template_folder = "templates", static_folder='static')

# Default route set as 'home'
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['GET'])
def predict_type():
    try:
        sepal_len = request.args.get('sep_len') # Get parameters for sepal length
        sepal_wid = request.args.get('sep_wid') # Get parameters for sepal width
        petal_len = request.args.get('pet_len') # Get parameters for petal length
        petal_wid = request.args.get('pet_wid') # Get parameters for petal width
        # Get the output from the classification model
        variety = model_new.predict(sepal_len, sepal_wid, petal_len, petal_wid)
        proba = model_new.predict_proba(sepal_len, sepal_wid, petal_len, petal_wid)
        if variety == 'Setosa':
            im = Image.open("img/setosa.jpg")
            data = io.BytesIO()
            im.save(data, "JPEG")
            encoded_img_data = base64.b64encode(data.getvalue())
        elif variety == 'Virginica':
            im = Image.open("img/virginica.jpg")
            data = io.BytesIO()
            im.save(data, "JPEG")
            encoded_img_data = base64.b64encode(data.getvalue())
        else:
            im = Image.open("img/versicolor.jpg")
            data = io.BytesIO()
            im.save(data, "JPEG")
            encoded_img_data = base64.b64encode(data.getvalue())
        #Render the output in new HTML page
        return render_template('home.html', variety = variety, img_data = encoded_img_data.decode('utf-8'), proba = proba)
    except:
        return 'Error'


if(__name__=='__main__'):
    app.run(port=5000)



