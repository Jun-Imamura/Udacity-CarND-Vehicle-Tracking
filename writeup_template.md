## Writeup

---

**Vehicle Detection Project**

#### The goals

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Estimate a bounding box for vehicles detected.

[image1]: ./output_images/Car_NotCar.png
[image2]: ./output_images/HOG.png
[image3]: ./output_images/features.png
[image4]: ./output_images/sliding_window.png
[image5]: ./output_images/DetectionResult.png
[image6]: ./output_images/heatmap.png
[image7]: ./output_images/final_res.png
[image7]: ./output_images/augment.png
[video1]: ./output.mp4

Here is the [Rubric](https://review.udacity.com/#!/rubrics/513/view) points for this project.

---
## Learn Vehicle Images Using Features and SVM

### 1. Histogram of Oriented Gradients (HOG)

The code for this step is contained in the first code cell of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

***

### 2. Final choice of HOG parameters.
I tried some combination of these parameters, and decided to use followings:
Color space was chosen by comparing SVN test set error.

```
orient = 10
pix_per_cell = 8
cell_per_block = 2
color_space = 'HSV'
```


***

### 3. Features Vector and Training

I trained a linear SVM using spatial feature, color histogram and HOG features for HSV color space.
Then, I did normalization. Below figure is the example feature vector for one image.

![alt text][image3]

When I applied SVM with default parameter, train accuracy was 1.0, which seems overfitting. So I decided to reduce the parameter `C` to avoid this (the parameter is related to decision boundary shape). And the parameter `C=0.00003` seems good to balance between train error and test error.

***

## Sliding Window Search

### 1. Sliding Window Search Algorithm and Parameter Exploration
I used multi-scale window sliding, with scaling image and fixed window size.
ROI is also chosen along with the scale size, because the vehicle in far region is captured upper (close to vanishing point) in the image plane.

```
scales = [1.0, 1.5, 2.0]
yranges = [(400, 500),(400, 550), (400, 600)]
```

![alt text][image4]


### 2.  images to demonstrate how your pipeline is working.

Here is the entire pipeline:

1. Crop ROI regions
2. Color conversion
3. Get spatial attributes (raw image resizing and vectorizing)
4. Get color histogram
5. Get HOG
6. Vectorize above 3 features and do normalization
7. Do prediction

Repeat 1-7 until all windows are examined.

![alt text][image5]

### 3. Corresponding Heatmap from Above Six Frames:
To reduce false positives, I utilized the thesholding in SVN output.
Even the windowed image is detected as <b>vehicle</b>, some might need to be ignored when the classification result is too close to decision boundary.

```
test_prediction = int(svc.decision_function(test_features) > svn_thresh)
```

And then, I used the idea of heatmapping. By incrementing each result of overwrapped sliding window, and then, thresholding is done to eliminate noisy detection.

Additionally, I introduced simple filtering algorithm to 
Below figure is the heatmap of detection result. This filtering is not included in the figure below (only applicable for sequencial images like video)

```
self.heatmap = self.heatmap // 2
self.heatmap = add_heat(heat, bboxes) // 2
```

![alt text][image6]

### 4. Here the resulting bounding boxes:
Based on heatmap, resulting bounding box is calculated as below:

![alt text][image7]


---

## Video Implementation

### 1. Process Against Sequencial Video
In the video sequence, we introduce `Heatmap` class for filtering against sequencial video. The reason is simply to apply filtering to reduce false positive detection.

```
class HeatMap():
    def __init__(self, img, threshold):        
        self.heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
        self.draw_img = np.zeros_like(img[:,:,0]).astype(np.float)
        self.threshold = threshold
    def update(self, img, bboxes):
        # Add heat to each box in box list        
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        self.heatmap = self.heatmap // 2
        self.heatmap = add_heat(heat, bboxes) // 2
        # Apply threshold to help remove false positives
        self.heatmap = apply_threshold(self.heatmap, self.threshold)
        # Find final boxes from heatmap using label function
        self.labels = label(self.heatmap)
    def get_heatmap(self):
        return self.heatmap
    def get_labels(self):
        return self.labels
    def get_draw_img(self):
        return self.draw_img
 ```


### 2. Filter for False Positives and Duplicated Detections

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here's a [link to my video result](./output.mp4)


---

### Discussion
It was not a easy task to create robust implementation. Here some method I utilized.

#### Data Augmentation
I did image augmentation to avoid overfitting. Here's the code which I used.

```python
import scipy.ndimage
def create_variant(image):
    if (random.choice([1, 0])):
        image = scipy.ndimage.interpolation.shift(input=image, shift=[random.randrange(-3, 3), random.randrange(-3, 3), 0])
    else:
        image = scipy.ndimage.interpolation.rotate(input=image, angle = random.randrange(-8, 8), reshape=False)
    return image
```


![alt text][image8]

#### Filtering Using Heatmap Class
In order to reject unstable detection result, I defined filtering method using previous detection result.
Basically, detected vehicles won't move so fast within the image coordinates, so the filtering doesn't have too much side-effect.

```python
class HeatMap():
    def __init__(self, img, threshold):        
        self.heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
        self.output = np.zeros_like(img[:,:,0]).astype(np.float)
        self.draw_img = np.zeros_like(img[:,:,0]).astype(np.float)
        self.threshold = threshold
        self.counter = 0
    def update(self, img, bboxes):
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        # Add heat to each box in box list 
        self.heatmap *= 0.5       
        self.heatmap += (add_heat(heat, bboxes) * 0.5)
        # Apply threshold to help remove false positives
        self.output = apply_threshold(self.heatmap, self.threshold)
        # Find final boxes from heatmap using label function
        self.labels = label(self.output)
```

#### Thresholding for SVN
Ouput of the SVN algorithm has the distance between hyperplane and data point.
So we can reject few likelihood data using thresholding.

```python
svn_thresh = 1.0
```


### Where my pipeline likely fails...
Light colored road surface tends to be mis-detected as vehicles.

### What could we do to make it more robust
It seems I need more data taken in various condition.