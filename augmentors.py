import cv2
import numpy as np
import os 

def crop_augmentor(train_x, train_y, preprocessed_path, num_per_class, aug_with='pos'):
    '''
    Generate new images by picking pairs of images and randomly replace region of one, quarter or half, by the same region of the other.

    # Example 
    ```python 
    x_aug,y_aug = crop_augmentor(train_x,train_y,1000, aug_with='pos')

    ```

    # Arguments
        train_x: numpy array with the names of images files.
        train_y: numpy array with labels of the images.
        num_per_class: the number of generated images per class.
        aug_with: if `pos` the pairs are of the same class. If `neg` the pairs of different classes. If `le` the pairs are of classes that are less or equal to the generated label.

    # Outputs
        train_x: numpy array with the base images names and the generated images names.
        train_y: numpy array with labels of the base images and the generated images.
    '''
    classnames = set(train_y)
    new_img_names = []
    new_img_classes = []
    for class_name in classnames:
        class_imgs1 = train_x[train_y == class_name]
        repeats = num_per_class//len(class_imgs1)
        if aug_with =='pos':
            class_imgs2 = class_imgs1.copy()
        elif aug_with =='neg':    
            class_imgs2 = train_x[train_y != class_name]
        elif aug_with =='le':
            class_imgs2 = train_x[train_y <= class_name]
        else: 
            raise ValueError('aug_with received unknown value.')
        for i in range(int(repeats)-1):
            np.random.shuffle(class_imgs2)
            for num in range(class_imgs1.shape[0]):
                img1 = cv2.imread(preprocessed_path+'train/'+ class_imgs1[num] +'.png')
                img2 = cv2.imread(preprocessed_path+'train/'+ class_imgs2[num] +'.png')
                if np.random.rand() > 0.5:
                    img1 = np.flipud(img1)
                if np.random.rand() > 0.5:
                    img1 = np.fliplr(img1)
                if np.random.rand() > 0.5:
                    img2 = np.flipud(img2)
                if np.random.rand() > 0.5:
                    img2 = np.fliplr(img2)
                img3 = img1.copy()
                chooser = 0.5
                x_corr = int(img3.shape[0]*(1 - (np.random.rand()*0.2+0.4)*(chooser < 0.6)))
                y_corr = int(img3.shape[1]*(1 - (np.random.rand()*0.2+0.4)*(chooser > 0.3)))
                img3[:x_corr,:y_corr,:] = img2[:x_corr,:y_corr,:]
                name = str(class_name) + str(i) + str(num) 
                cv2.imwrite(preprocessed_path+'train/'+name+'.png', img3)
                new_img_names.append(name)
                new_img_classes.append(class_name)
        np.random.shuffle(class_imgs2)
        np.random.shuffle(class_imgs1)
        for num in range(int(num_per_class - repeats * len(class_imgs1))):
            img1 = cv2.imread(preprocessed_path+'train/'+ class_imgs1[num] +'.png')
            img2 = cv2.imread(preprocessed_path+'train/'+ class_imgs2[num] +'.png')
            if np.random.rand() > 0.5:
                img1 = np.flipud(img1)
            if np.random.rand() > 0.5:
                img1 = np.fliplr(img1)
            if np.random.rand() > 0.5:
                img2 = np.flipud(img2)
            if np.random.rand() > 0.5:
                img2 = np.fliplr(img2)
            img3 = img1.copy()
            chooser = 0.5
            x_corr = int(img3.shape[0]*(1 - (np.random.rand()*0.2+0.4)*(chooser < 0.6)))
            y_corr = int(img3.shape[1]*(1 - (np.random.rand()*0.2+0.4)*(chooser > 0.3)))
            img3[:x_corr,:y_corr,:] = img2[:x_corr,:y_corr,:]
            name = str(class_name) + str(num) + 're' 
            cv2.imwrite(preprocessed_path+'train/'+name+'.png', img3)
            new_img_names.append(name)
            new_img_classes.append(class_name)
    train_y = np.append(train_y, new_img_classes, axis=0)
    train_x = np.append(train_x, new_img_names, axis=0)
    return train_x, train_y

