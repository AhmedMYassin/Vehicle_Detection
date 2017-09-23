# Vehicle Detection Project

In Vehicle Detection Project, it's expected to be able to detect vehicle in images and apply this detection algorithm to videos to track vehicles.
To detect vehicles in images, I needed to perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear support vector machine classifier

[//]: # (Image References)
[image1]: ./output_images/Vehicle_and_Non_Vehicle.png
[image2]: ./output_images/HOG0.png
[image4]: ./output_images/HOG1.png
[image5]: ./output_images/HOG2.png
[image6]: ./output_images/Test_Sliding_Windows.png
[image7]: ./output_images/Video_Test_Heat_map.png
[image8]: ./output_images/Video_Test_Heat_map1.png
[image9]: ./output_images/Video_Test_Heat_map2.png
[image10]: ./output_images/Video_Test_Heat_map3.png
[image11]: ./output_images/Video_Test_Heat_map4.png
[image12]: ./output_images/Video_Test_Heat_map5.png
[image13]: ./output_images/Video_Test_Output.png
[video1]: ./project_video.mp4


### Histogram of Oriented Gradients (HOG)

In IPython notebook, I start with getting my training data ""Kitti dataset for vehicle and non-vehicle images".  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I had explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I used the random images to get a feel for what the `skimage.hog()` output looks like. 

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]
![alt text][image4]
![alt text][image5]

I had also computed the spatial features of size `(32,32)` and the color histogram features for `32` bins.

I used the final parameters of HOG to extract the HOG features of the training data to use them beside the color histogram features and the spatial features as input to the classifier. I made sure that the images are normalized `(x - 0.5)`.

I trained a linear SVM using 'grid_search.GridSearchCV()' and the final calssification paramters were `kernel = rbf` and `C = 1.0`. The classifier accuracy was 0.995 over the testing data .. maybe this accuracy isn't reliable as the difference between the training and testing data isn't big enough to trust this accuracy.


### Sliding Window Search

I used sliding windows of size multiple of `32x32` to cover the range from `64x64` to `256x256` using 25% of overlapping.

I performed the sliding window search on test images to create different windows, then I extracted HOG features from these windows as mentioned above and apply the classifier on them. Test images were normalized.

To avoid false positives, I used heatmap to filter windows based on threshold = 1. Here you can check the test images after removing the false positives.

![alt text][image6]


### Video Implementation

[Here](./project_video.mp4) you can find the output video.

To filter the flase positives, I used the positive detections in batches of consecutive frames to create a heatmap which was thresholded to get rid of all false positives.

Here's an example result showing the heatmap from a series of frames of video.

Here are 5 frames and their corresponding heatmaps:

![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]

Here is the resulting bounding boxes in last frame and the output of`scipy.ndimage.measurements.label()` on the integrated heatmap from all 5 frames:
![alt text][image13]

### Discussion

The pipeline in video implementation isn't enough to avoid whobly boxes. Using batches of consecutive frames made the result much better but it's not the best way to do that. Maybe a specific tracking algorithm is needed for that.
