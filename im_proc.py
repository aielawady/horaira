def circle_centering(image, circle_detection_method='moments',
                     scale_value = 1, aspect_ratio=1.6, width=224, gray_threshold = 10):
    ''' 
    Applies Affine Transformation to move and scale the circle to the center of the image (diabetic retinopathy) 
    
    # Arguments
    image: numpy array
    circle_detection_method: `moments`, `enclosing_circle` or `max_dim`. `moments` to get the centroid of the binary image obtained by binary comparison of the gray image with gray_threshold and the radius of a circle having area equivalent to that of the binary image. `enclosing_circle` detects the enclosing circle of the largest contour and uses it's center and radius. `max_dim` uses the max dimension of the image as the raduis and the center of the image as the center
    scale_value: float value by which the detected circle is scaled.
    aspect_ratio: aspect ratio for crop.
    width: the width of the resulting image.
    gray_threshold: int used to separate the eye from the background.
    
    Output
    image: numpy array contains the circle detected by the `method` in the center and having a raduis of the `width`.
    '''
    
    while image.shape[1] < width:  
        image = cv2.resize(image,(image.shape[1]*2,image.shape[0]*2))
        
    x_bi = (cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) > gray_threshold).astype('uint8')*128

    if circle_detection_method == 'enclosing_circle':
        contours,hierarchy = cv2.findContours(x_bi, 1, 2)
        cnt = contours[np.argmax(np.array([len(image) for image in contours]))]
        (cx,cy),radius = cv2.minEnclosingCircle(cnt)
        center = [int(cx),int(cy)]
        radius = int(radius)
    elif circle_detection_method == 'moments':
        M = cv2.moments(x_bi)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        center = [int(cX), int(cY)]
        radius = int(np.sqrt(M['m00']/np.pi/128))
    elif circle_detection_method == 'max_dim':
        m,n = x_bi.shape
        center = [n//2,m//2]
        radius = max([m//2, n//2])
    flpup = 1
    flplft = 1
    shift_val_x = 0
    shift_val_y = 0
    rotation_val = 0
    scale_val = scale_value
    center_new = [width//2 + shift_val_x,width//2 + shift_val_y]
    pnt1 = [int(center_new[0] - flplft * np.sin(rotation_val) * (width/2 * scale_val)), int(center_new[1] + flpup * np.cos(rotation_val) * (width/2 * scale_val))]
    pnt2 = [int(center_new[0] + flplft * np.cos(rotation_val) * (width/2 * scale_val)), int(center_new[1] + flpup * np.sin(rotation_val) * (width/2 * scale_val))]
    tri1 = np.float32([[center, [center[0],center[1] + radius], [center[0] + radius,center[1]]]])
    tri2 = np.float32([[center_new, pnt1, pnt2]])
    M = cv2.getAffineTransform(tri1,tri2)
    image = cv2.warpAffine(image, M, (width,width))
    image = image[int(width/2*(1-1/aspect_ratio)):int(width/aspect_ratio+width/2*(1-1/aspect_ratio)), :int(width), :]
    return image

def background_noiser(img, gray_threshold=10):
    '''
    Adds noise to regions of the image with a value less than the threshold.
    
    # Arguments
    img: numpy array of the image.
    gray_threshold: gray threshold.

    # Outputs
    img: the image with noise over the regions that have gray values less than gray_threshold.   
    '''
    tmp_c = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)<gray_threshold
    img[tmp_c,:] = (np.random.normal(size=3*np.sum(tmp_c))).reshape(int(np.sum(tmp_c)),3)*255
    return img

def pyramid_scale_down(img, width=300):
    if img.shape[1] > 2*(0.9*width):
        img = cv2.pyrDown(img)
    return img

def veins_spots_highlighter(src,ddepth = cv2.CV_16S,kernel_size = 3):
    src = cv2.GaussianBlur(src, (5, 5), 0)
    src_gray = (cv2.cvtColor(src, cv2.COLOR_BGR2GRAY))
    dst = cv2.Laplacian(src_gray, ddepth, ksize=5)
    dst = cv2.medianBlur(dst,3)
    abs_dst = cv2.convertScaleAbs(dst)
    circle_img = np.zeros(src.shape[:2], np.uint8)
    cv2.circle(circle_img,(int(src.shape[1]/2), int(src.shape[0]/2)),int(src.shape[1]/2*0.99),1,thickness=-1)
    masked_data = cv2.bitwise_and(abs_dst, abs_dst, mask=circle_img)
    masked_data = cv2.GaussianBlur(masked_data, (21, 21), 0)
    masked_data = cv2.equalizeHist(masked_data) > 150
    mock = cv2.cvtColor(src,cv2.COLOR_BGR2HLS)
    mock[:,:,1] = cv2.bitwise_and(mock[:,:,1], mock[:,:,1], mask=masked_data.astype('uint8'))
    mock = cv2.cvtColor(mock, cv2.COLOR_HLS2BGR)
    return mock

def edger(img, gray_threshold=15, kernel_size=5,dilate_itr=1):
    msk = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)<gray_threshold
    kernel_size=img.shape[1]//100
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    kernel_size_blur=img.shape[1]//5
    kernel_blur = np.ones((kernel_size,kernel_size),np.uint8)
    msk_blur = cv2.morphologyEx(msk.astype('uint8'), cv2.MORPH_OPEN, kernel_blur).astype(bool)
    msk = cv2.morphologyEx(msk.astype('uint8'), cv2.MORPH_OPEN, kernel)
    msk = cv2.dilate(msk.astype('uint8'),kernel,iterations = dilate_itr).astype(bool)
    img[msk] = np.mean(np.mean(img,axis=0),axis=0)
    msk_smoother = np.expand_dims(cv2.GaussianBlur(msk.astype(float),(img.shape[1]//20-(img.shape[1]//20)%2+1,img.shape[1]//20-(img.shape[1]//20)%2+1),img.shape[1]//10),axis=2)
    img = msk_smoother*cv2.GaussianBlur(img,(img.shape[1]//10-(img.shape[1]//10)%2+1,img.shape[1]//10-(img.shape[1]//10)%2+1),img.shape[1]//1) + (1-msk_smoother)*img
    return img.astype('float32')
