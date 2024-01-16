from flask import Flask, jsonify, render_template, request
import numpy as np
import make_prediction
import onto
from onto import SPARQL, IndividualGenerator  

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_path = data.get('image_path')
    longitude = float(data.get('longitude'))
    latitude = float(data.get('latitude'))

    test_image_paths = make_prediction.get_dataset_slice_paths('croped_images')
    model = make_prediction.segmentation_model()
    model.load_weights('model_third_parameteres.h5')

    for img in test_image_paths:
        test_dataset = make_prediction.get_test_data(img)
        prediction = model.predict(test_dataset)
        prediction = np.argmax(prediction, axis=3)
        
        onto.create_image_individual(image_path, longitude, latitude)
        x_coordinate = img.split("-")[1]
        y_coordinate = img.split("-")[2]
        predicted_image = prediction[0]
        onto.create_terrene_individuals(predicted_image, img, x_coordinate, y_coordinate.split(".")[0])

    return jsonify({'message': 'Prediction completed successfully'})


@app.route('/get_individuals_form', methods=['GET'])
def get_individuals_form():
    return render_template('get_individuals_form.html')


@app.route('/get_individuals', methods=['GET'])
def get_individuals():
    try:
        image_name = request.args.get('image_name', type=str)
        class_name = request.args.get('class_name', type=str)
        x1 = request.args.get('x1', type=int, default=0)
        y1 = request.args.get('y1', type=int, default=0)
        x2 = request.args.get('x2', type=int, default=1e6)
        y2 = request.args.get('y2', type=int, default=1e6)
        min_percentage = request.args.get('min_percentage', type=float, default=0)
        max_percentage = request.args.get('max_percentage', type=float, default=100)

        individuals = SPARQL.filter_individuals(
            image_name=image_name,
            class_name=class_name,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            min_percentage=min_percentage,
            max_percentage=max_percentage
        )
        
        results = []
        for ind in individuals:
            result = {
                'class': ind[0].split("_")[-1],
                'x': ind[1],
                'y': ind[2],
                'percentage': ind[3]
            }
            results.append(result)
        if len(results) == 0:
            return jsonify({'message': 'No individuals found'})
   

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5500)


# curl --noproxy 127.0.0.1 -X POST -H "Content-Type: application/json" -d '{"image_path": "GP3_orto_A3.tif", "longitude": 42.3, "latitude": 22.4}' http://127.0.0.1:5500/predict

# curl --noproxy 127.0.0.1 -X GET -H "Content-Type: application/json" -d '{"image_name": "GP3_orto_A3.tif", "class_name": "your_class_name", "x1": 0, "y1": 0, "x2": 100, "y2": 100, "min_percentage": 0, "max_percentage": 100}' http://127.0.0.1:5500/get_individuals
