## ECO-ontology

In this project, terrain segmentation was performed based on the plant species present in the area. The images were captured by aerial photography. The images are in TIFF format and have large dimensions and resolutions.

The processing begins with loading the TIFF image, followed by a preprocessing step where the large image is cut into dimensions of 512x512 pixels. This results in several smaller images from one large image, each subjected to predictions. Machine learning methods, using the VGG model, are employed for prediction. The input to the model is a cut image, and the output is the same image with masks. The model determines the class to which the image belongs, and based on the returned mask, the percentage of the detected plant species is calculated.

An ontology is generated based on the obtained information, creating individuals that can be queried using SPARQL. Queries on the ontology can be executed from a web application. The third processing step is illustrated in the figure:

![alt text](https://github.com/YoNoSoyMarinero/EnvoOntology/blob/main/readme/web%20arch.drawio.png?raw=true)

#### Dataset
The dataset consists of a few images, each with large dimensions and resolution. These images were captured over a specific marshy area from an aircraft. Machine learning algorithms are applied to obtain information about the prevalence of the plant community in the surveyed area. The ontology is generated based on the obtained information, but more details are provided in the ontology section. For more efficient processing, the initial image is cropped to dimensions of 512x512. The resulting images are annotated using the [via-2.0.12](https://www.robots.ox.ac.uk/~vgg/software/via/) annotation tool and then fed to the model for processing.

#### Model
To determine the presence of a plant community in a specific area, the VGG16 machine learning model is used on the image.

[VGG16](https://arxiv.org/pdf/1409.1556.pdf) is a machine learning model designed for image processing, notable for winning the ImageNet 2014 competition in classification and object detection. With approximately 138 million parameters, it achieves an accuracy of 92.7% across 1000 classes. VGG16, with 16 layers, is commonly used for image classification and is easily applicable and suitable for additional training. A special feature of this model is the use of small 3x3 filters in convolutional layers, resulting in significant improvement over previous architectures. The VGG16 architecture includes 13 convolutional layers, five max-pooling layers, and three fully connected layers.

# SLIKA arhitekture modela dodati

After the model makes predictions, it returns detected classes for the submitted image. Based on these predictions, visualizing detections is possible, as shown in the figure. 
# SLIKA detekcije ako izgleda maska dodati


#### Ontology
Based on the detection results, it is possible to determine the plant community represented in the image, the percentage of the plant community, and coordinates. Coordinates can be calculated based on the displacement during image cropping. Using this information and the image name, an ontology with individuals is generated. Owlready2 is used as a basis for classes, from which specific classes are taken to build the ontology. The following classes are used: 
```Terrene, Image, Empty, Vegetation, Soil, Liquid, TreeCanopy, Water, Rock, Dirt, Waterlilly, Swamp, Grass, Sand```. When the system is first run, a file is generated to store the ontology and individual definitions. Each subsequent run appends newly generated individuals to the end of the file.


#### Web app
To facilitate querying the generated individuals, a website has been developed with a user-friendly graphical interface. Users can submit queries using a form, and the website also allows image submission for prediction.

# SLIKA aplikacije dodati

#### SPARQL Query
SPARQL is a query language used for working with RDF (Resource Description Framework) data. RDF is a standard model for data exchange on the web, often used to describe resources and their properties. SPARQL allows you to query RDF data to extract relevant information. This simple query allows searching for individuals based on the original image name, while other parameters are optional and can be entered by the user. In that case, the query returns all individuals related to the submitted image name. Additionally, it is possible to modify the class to which the plant community belongs, the coordinates of the search area, and the minimum and maximum percentage of representation of that plant species.

### Running the System
To run the system, follow these steps:

Activate the virtual environment:
```
python3 -m venv ime_venv

# Windows
ime_venv\Scripts\activate 

# Linux
source ime_venv/bin/activate
```
Install the required libraries listed in the requirements.txt file:
```
pip3 install -r requirements.txt
```

After completing these steps, you can start the system by running the ```app.py``` script:

```
python3 app.py
```

##### Notes
If you encounter problems running the model on your GPU, add the following line of code to the make_prediction.py file to disable GPU usage:
```
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
``````
