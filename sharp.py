import cv2

image = cv2.imread("./image/inception.jpg") # input unsharpen image
blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
unsharped_img = cv2.addWeighted(image, 2.0, blurred, -1.0, 0)
rgb_results = cv2.cvtColor(unsharped_img, cv2.COLOR_BGR2RGB)
cv2.imshow('original image', image)
cv2.imshow('sharpened image',unsharped_img)
cv2.imwrite("./sharpened_image/inception.jpg", unsharped_img) # output sharpened image
cv2.waitKey(0)