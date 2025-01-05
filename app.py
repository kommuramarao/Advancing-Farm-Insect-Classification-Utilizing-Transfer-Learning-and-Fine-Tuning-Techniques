from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Folder where uploaded images are saved
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
# Load the SavedModel format
# model = tf.keras.models.load_model('model/insect_model')
# model = tf.keras.models.load_model('model/insect_model.h5')
model = tf.keras.models.load_model('model/insect_model.keras')




# model = tf.keras.models.load_model('C:/Users/manee/Downloads/main-codes_new/main-codes_new/farm_insect_classifier/model/insect_model.h5')
# file_path = "C:/Users/manee/Downloads/main-codes_new/main-codes_new/farm_insect_classifier/model/insect_model.h5"

# Define the class names (ensure it matches the order used during training)
class_names = ['Africanized Honey Bees (Killer Bees)', 'Aphids', 'Armyworms', 'Brown Marmorated Stink Bugs', 'Cabbage Loopers', 'Citrus Canker', 'Colorado Potato Beetles', 'Corn Borers', 'Corn Earworms', 'Fall Armyworms', 'Fruit Flies', 'Spider Mites', 'Thrips', 'Tomato Hornworms', 'Western Corn Rootworms']  # Replace with your actual class names

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess the image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict using the model
        predictions = model.predict([img_array, img_array])  # Since your model expects two inputs
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_names[predicted_class]

        # Render the result with the image
        return render_template('result.html', prediction=predicted_label, image_file=filename)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
