**Vehicle Detection Project**

In Vehicle Detection Project, it's expected to be able to detect vehicle in images and apply this detection algorithm to videos to track vehicles.
To detect vehicles in images, I needed to perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear support vector machine classifier

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

### Histogram of Oriented Gradients (HOG)

#### 1. In IPython notebook, I start with getting my training data ""Kitti dataset for vehicle and non-vehicle images".  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

#### 2. I had explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I used the random images to get a feel for what the `skimage.hog()` output looks like. 

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 3. I used the final parameters to extract the features of the training data to use them as input for the classifier. I made sure that the images are normalized.

#### 4. I trained a linear SVM using 'grid_search.GridSearchCV()' and the final calssification paramters were 'kernel = rbf' and 'C = 1.0'. The classifier accuracy was 0.995 over the testing data .. maybe this accuracy isn't reliable as the difference between the training and testing data isn't big enough to trust this accuracy.

### Sliding Window Search

#### 1. I used sliding windows of size multiple of 32x32 to cover the range from 64*64 to 256*256 using 25% of overlapping.

#### 2. I performed the sliding window search on test images to create different windows, then I extracted HOG features from these windows as mentioned above and apply the classifier on them. Test images were normalized.

#### 3. To avoid false positives, I used heatmap to filter windows based on threshold = 1. Here you can check the test images after removing the false positives.

![alt text][image4]
---

### Video Implementation

####1. Here's a [link to my video result](./project_video.mp4)


####2. To filter the flase positives, I used the positive detections in batches of concecutive frames to create a heatmap which was thresholded to get rid of all false positives.

Here's an example result showing the heatmap from a series of frames of video.

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

###Discussion

#### The pipeline in video implementation isn't enough to avoid whobly boxes. Using batches of consecutive frames made the result much better but it's not the best way to do that. Maybe a specific tracking algorithm is needed for that.
