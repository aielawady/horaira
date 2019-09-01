import cv2
import numpy as np



def apply_preprocess(names,src_path,dst_path=None,extension='png',preprocessing_function=[],
                     preprocessing_params=[], write=False, prog_bar_disable=False):
    '''
    Applys list of functions to list of images specified by name and path
    
    # Example

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

    # Arguments
    names: list of image names
    src_path: source path
    dst_path: destination path
    extension: image extension
    preprocessing_function: list of functions to apply in order on the image
    preprocessing_params: list of dictionaries of the same length as preprocessing_function that contain the parameter for each function in order.
    write: if `True` the resulting images will be written in the dst_path. If `False` the resultingimages are returned.
    prog_bar_disable: if `False` the function will show the progress bar. If `True` the function will not show the progress bar.
    
    # Outputs
    Returns the resuling images if `write==False` and returns and empty list if `write==True`.

    '''

    assert(len(preprocessing_function) == len(preprocessing_params))
    assert(os.path.isdir(src_path))
    if src_path[-1]!='/':
        src_path.append('/')
    if write:
        assert(os.path.isdir(dst_path))
        if dst_path[-1]!='/':
            dst_path.append('/')
    
    img_list = []
    for name in tqdm(names, disable=prog_bar_disable):
        if extension == 'npy':
            img = np.load(src_path+name+'.'+extension)
        else:
            img = load_img(src_path+name+'.'+extension)
            img = img_to_array(img)
        for func,params in zip(preprocessing_function,preprocessing_params):
            img = func(img,**params)
        if write:
#             cv2.imwrite(dst_path+name+'.'+extension, img)
            np.save(dst_path+name, img.astype('uint8'))
        else:
            img_list.append(img)
    return img_list

def balanced_valid_set_splitter(X,y, classes=None,valid_n_per_class=30,shuffle=False, debug=False):
    '''
    Splits the dataset to a training set and validation set. The resulting validation set is balanced.

    # Example

    # Arguments
    X: numpy array of the examples.
    y: numpy array of the labels.
    classes: list of labels the dataset. If `None` is provided,`classes` will be set as follows `classes = np.unique(y, axis=0).
    valid_n_per_class: Number of examples per class in the validation set.
    shuffle: shuffles the dataset if `shuffle==True`.
    debug: displays histograms of the training set and the validation set if set to `True`.

    # Outputs
    train_x: numpy array of the training examples.
    train_y: numpy array of the training labels.
    valid_x: numpy array of the validation examples.
    valid_y: numpy array of the validation labels.

    '''

    train_x = np.array([])
    train_y = np.array([])
    valid_x = np.array([])
    valid_y = np.array([])
    
    if not classes:
        classes = np.unique(y, axis=0)
    
    if shuffle:
        indexes = np.random.randint(classes.shape[0],size=(classes.shape[0]))
        X = X[indexes]
        y = y[indexes]
        
    for i in classes:
        valid_x = np.append(valid_x,X[y==i][:valid_n_per_class],axis=0)
        valid_y = np.append(valid_y,y[y==i][:valid_n_per_class],axis=0)
        train_x = np.append(train_x,X[y==i][valid_n_per_class:],axis=0)    
        train_y = np.append(train_y,y[y==i][valid_n_per_class:],axis=0)  
    
    if debug:
        plt.subplot(2,1,1)
        plt.hist(valid_y)
        plt.subplot(2,1,2)
        plt.hist(train_y)
        
    return train_x,train_y,valid_x,valid_y
    
def classes_encoder(y,n_classes,method='one'):
    '''
    Encodes the labels of the dataset.  

    # Arguments
    y: numpy array of the classes.
    n_classes: number of classes in y.
    method: if `one` the examples are encoded in one-hot encoding. If `all_lower_ones` the examples are encoded as a multilabel classes that have all lower classes as labels. If `pairs` the examples are encoded as a multilabel classes that have the labels of it's class and the next lower class.

    # Outputs
    y_coded: numpy array contains the encoded `y`s.
    '''
    if method=='one':
        return to_categorical(y, num_classes=n_classes)
    elif method=='all_lower_ones':
        y_coded = np.ones((len(y), n_classes))
        mask = np.repeat(np.arange(5).reshape(1,-1),len(y),axis=0) 
        mask = mask > y.reshape(-1,1)
        y_coded[mask] = 0
        return y_coded
    elif method=='pairs':
        y_coded = np.zeros((len(y), n_classes))
        mask = np.repeat(np.arange(n_classes).reshape(1,-1),len(y),axis=0) 
        mask = (mask == y.reshape(-1,1)) | (mask == y.reshape(-1,1)-1)
        y_coded[mask] = 1
        return y_coded
    
    
def classes_decoder(y,method='max'):
    '''
    Decodes the results of classification problem.

    # Arguments
    y: numpy array of the prediction from a classification NN of shape `(number of examples, number of classes)`.
    method: if `max` the examples are classified based on max value. If `highest_true` the examples are classified based on the highest true class as if the examples have multilabels.

    # Outputs
    numpy array with the classification.

    '''
    n_classes = y.shape[1]
    if method=='max':
        return np.argmax(y,axis=1)
    elif method=='highest_true':
        masked_array = np.arange(n_classes)*(y>=0.5).astype('int')
        return np.argmax(masked_array,axis=1)
