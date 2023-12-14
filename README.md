The project focuses on using visual computing techniques to recognize the food components inside a image of daily meal. By comparatively studying on two image classification models and one object detection model, YOLOv8 is selected as the most suitable training model for our data set, for its high detection accuracy and ability of boxing the food components. Then a nutrition advice on the input meal image is given based on the model result.

<h2>Results</h2>
<p>Feeding a food image to the model, we get output as a predicted category and its likelihood, as well as a bounding box in pixel coordinates for each detected food item. Based on these, we then generate a piece of nutritional advice indicating the lack or overeating of certain categories.</p>

<h3>Example Output</h3>
<p align='left'>
  <img src="https://github.com/tyrrr110/FoodItemDetect/assets/49021993/bc6e62d7-3428-491c-915e-3fab21e9606f">
</p>

> [{'Category': 'Carbohydrates', 'bbox': [448.82, 0.0, 186.943, 172.124], 'score': 0.92466},  
  {'Category': 'Vegetables', 'bbox': [374.734, 170.676, 244.255, 235.215], 'score': 0.78086},  
  {'Category': 'Vegetables', 'bbox': [200.128, 0.0, 187.217, 215.673], 'score': 0.45619},  
  {'Category': 'Cheese', 'bbox': [453.453, 51.224, 112.915, 105.642], 'score': 0.43432}]
