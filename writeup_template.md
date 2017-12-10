## Writeup

---

**Vehicle Detection Project**

#### The goals

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Estimate a bounding box for vehicles detected.

[image1]: ./output_images/Car_NotCar.png
[image2]: ./output_images/HOG.png
[image3]: ./output_images/features.png
[image4]: ./output_images/DetectionResult.png
[image5]: ./output_images/heatmap.png
[image6]: ./output_images/final_res.png
[video1]: ./project_video.mp4

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
I chose following HOG parameters

```
orient = 9
pix_per_cell = 8
cell_per_block = 2
```

***

### 3. Features Vector and Training

I trained a linear SVM using spatial feature, color histogram and HOG features for HSV color space.
Then, I did normalization. Below figure is the example feature vector for one image.

![alt text][image3]

***

## Sliding Window Search

### 1. Sliding Window Search Algorithm and Parameter Exploration
I scaled input image to $1/1.5$, for optimization. And window size is fixed to 64.



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

![alt text][image4]

### 3. Corresponding Heatmap from Above Six Frames:
To reduce false positives, and integrate multiple detection, the idea of heatmap is introduced. In the heatmap, each detected result of bounding boxes are incremented, and then, thresholding is done to eliminate noisy detection.

Below figure is the heatmap of detection result.

![alt text][image5]

### 4. Here the resulting bounding boxes:
Based on heatmap, resulting bounding box is calculated as below:

![alt text][image6]


---

## Video Implementation

### 1. Process Against Sequencial Video
<b>Sorry, still under implementation</b>

In the video sequence, we introduce `Heatmap` class for filtering against sequencial video.


### 2. Filter for False Positives and Duplicated Detections

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here's a [link to my video result](./project_video.mp4)


---

### Discussion

Now the problem is:

* Shadows are mis-detected as vehicles.
* Bounding box for the vehicle is not stable.
