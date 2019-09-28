# Namespace `horaira` {#horaira}





    
## Sub-modules

* [horaira.augmentors](#horaira.augmentors)
* [horaira.im_proc](#horaira.im_proc)
* [horaira.utils](#horaira.utils)






    
# Module `horaira.augmentors` {#horaira.augmentors}







    
## Functions


    
### Function `crop_augmentor` {#horaira.augmentors.crop_augmentor}



    
> `def crop_augmentor(train_x, train_y, num_per_class, aug_with='pos')`


Generate new images by picking pairs of images and randomly replace region of one, quarter or half, by the same region of the other.

##### Example 
```python 
x_aug,y_aug = crop_augmentor(train_x,train_y,1000, aug_with='pos')

```

##### Arguments
    train_x: numpy array with the names of images files.
    train_y: numpy array with labels of the images.
    num_per_class: the number of generated images per class.
    aug_with: if `pos` the pairs are of the same class. If `neg` the pairs of different classes. If `le` the pairs are of classes that are less or equal to the generated label.

##### Outputs
    train_x: numpy array with the base images names and the generated images names.
    train_y: numpy array with labels of the base images and the generated images.





    
# Module `horaira.im_proc` {#horaira.im_proc}







    
## Functions


    
### Function `background_noiser` {#horaira.im_proc.background_noiser}



    
> `def background_noiser(img, gray_threshold=10)`


Adds noise to regions of the image with a value less than the threshold.

##### Arguments
img: numpy array of the image.
gray_threshold: gray threshold.

##### Outputs
img: the image with noise over the regions that have gray values less than gray_threshold.


    
### Function `circle_centering` {#horaira.im_proc.circle_centering}



    
> `def circle_centering(image, circle_detection_method='moments', scale_value=1, aspect_ratio=1.6, width=224, gray_threshold=10)`


Applies Affine Transformation to move and scale the circle to the center of the image (diabetic retinopathy) 

##### Arguments
image: numpy array
circle_detection_method: `moments`, `enclosing_circle` or `max_dim`. `moments` to get the centroid of the binary image obtained by binary comparison of the gray image with gray_threshold and the radius of a circle having area equivalent to that of the binary image. `enclosing_circle` detects the enclosing circle of the largest contour and uses it's center and radius. `max_dim` uses the max dimension of the image as the raduis and the center of the image as the center
scale_value: float value by which the detected circle is scaled.
aspect_ratio: aspect ratio for crop.
width: the width of the resulting image.
gray_threshold: int used to separate the eye from the background.

Output
image: numpy array contains the circle detected by the `method` in the center and having a raduis of the `width`.


    
### Function `edger` {#horaira.im_proc.edger}



    
> `def edger(img, gray_threshold=15, kernel_size=5, dilate_itr=1)`





    
### Function `pyramid_scale_down` {#horaira.im_proc.pyramid_scale_down}



    
> `def pyramid_scale_down(img, width=300)`





    
### Function `veins_spots_highlighter` {#horaira.im_proc.veins_spots_highlighter}



    
> `def veins_spots_highlighter(src, ddepth=3, kernel_size=3)`








    
# Module `horaira.utils` {#horaira.utils}







    
## Functions


    
### Function `apply_preprocess` {#horaira.utils.apply_preprocess}



    
> `def apply_preprocess(names, src_path, dst_path=None, extension='png', preprocessing_function=[], preprocessing_params=[], write=False, prog_bar_disable=False)`


Applys list of functions to list of images specified by name and path

##### Example

```python
circle_centering_params = {
    'circle_detection_method':'enclosing_circle',
    'scale_value' : scale_value,
    'aspect_ratio' : aspect_ratio,
    'width' : image_size, 
    'gray_threshold' : 12
}
preprocess_sequence = [circle_centering]
preprocess_params = [circle_centering_params]
lister = np.random.choice(df_train['id_code'],25)
imgs = apply_preprocess(lister,src_path=preprocessed_path+'train/', preprocessing_function=[preprocess_sequence], preprocessing_params=[preprocess_params])
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(imgs[i])
```

##### Arguments
names: list of image names
src_path: source path
dst_path: destination path
extension: image extension
preprocessing_function: list of functions to apply in order on the image
preprocessing_params: list of dictionaries of the same length as preprocessing_function that contain the parameter for each function in order.
write: if `True` the resulting images will be written in the dst_path. If `False` the resultingimages are returned.
prog_bar_disable: if `False` the function will show the progress bar. If `True` the function will not show the progress bar.

##### Outputs
Returns the resuling images if `write==False` and returns and empty list if `write==True`.


    
### Function `balanced_valid_set_splitter` {#horaira.utils.balanced_valid_set_splitter}



    
> `def balanced_valid_set_splitter(X, y, classes=None, valid_n_per_class=30, shuffle=False, debug=False)`


Splits the dataset to a training set and validation set. The resulting validation set is balanced.

##### Example

##### Arguments
X: numpy array of the examples.
y: numpy array of the labels.
classes: list of labels the dataset. If `None` is provided,`classes` will be set as follows `classes = np.unique(y, axis=0).
valid_n_per_class: Number of examples per class in the validation set.
shuffle: shuffles the dataset if `shuffle==True`.
debug: displays histograms of the training set and the validation set if set to `True`.

##### Outputs
train_x: numpy array of the training examples.
train_y: numpy array of the training labels.
valid_x: numpy array of the validation examples.
valid_y: numpy array of the validation labels.


    
### Function `classes_decoder` {#horaira.utils.classes_decoder}



    
> `def classes_decoder(y, method='max')`


Decodes the results of classification problem.

##### Arguments
y: numpy array of the prediction from a classification NN of shape `(number of examples, number of classes)`.
method: if `max` the examples are classified based on max value. If `highest_true` the examples are classified based on the highest true class as if the examples have multilabels.

##### Outputs
numpy array with the classification.


    
### Function `classes_encoder` {#horaira.utils.classes_encoder}



    
> `def classes_encoder(y, n_classes, method='one')`


Encodes the labels of the dataset.  

##### Arguments
y: numpy array of the classes.
n_classes: number of classes in y.
method: if `one` the examples are encoded in one-hot encoding. If `all_lower_ones` the examples are encoded as a multilabel classes that have all lower classes as labels. If `pairs` the examples are encoded as a multilabel classes that have the labels of it's class and the next lower class.

##### Outputs
y_coded: numpy array contains the encoded `y`s.


