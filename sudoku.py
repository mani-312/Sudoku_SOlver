# Importing necessary libraries
import cv2
import numpy as np
from imutils.perspective import four_point_transform
from tensorflow.keras.preprocessing.image import img_to_array
from skimage.segmentation import clear_border
from tensorflow.keras.models import load_model
import imutils

# Finds the digit component in the image
def connected(gray):
    blurred = cv2.GaussianBlur(gray,(7,7),3)
    (T,thresh) = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    output = cv2.connectedComponentsWithStats(
              #connectivity  #dtype
	thresh,   4,        cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    for i in range(numLabels):
        x = stats[i,0]
        y = stats[i,1]
        w = stats[i,2]
        h = stats[i,3]
        area = stats[i,4]
        (cX,cY) = centroids[i]
        if(w==0 and h==0):
            continue

        # Handpicked these values for getting digits
        keepWidth = w > 7 and w < 25
        keepHeight = h > 14 and h < 28
        keepArea = area > 100 and area < 370
        if((keepWidth & keepHeight & keepArea)):
            component = np.zeros(gray.shape,dtype='uint8')
            component[y:y+h,x:x+w] = gray[y:y+h,x:x+w]
            return component
    return None

# Checking whether we can place matrix[i,j] = val
def check(matrix,i,j,val):
    for k in range(9):
        if(matrix[i,k]==val or matrix[k,j]==val):
            return 0;
    x = i-(i%3)
    y = j-(j%3)
    for i in range(x,x+3):
        for j in range(y,y+3):
            if(matrix[i,j]==val):
                return 0

    return 1

# Finds the solution to the sudoku grid
def solve(matrix,i,j):
    if(i==9):
        return (matrix,1)
    elif(j==9):
        return solve(matrix,i+1,0)
    elif(matrix[i,j]!=0):
        return solve(matrix,i,j+1)
    for val in range(1,10):
        if(check(matrix,i,j,val)):
            matrix[i,j]=val
            (output,flag) = solve(matrix,i,j+1)
            if(flag==1):
                return (output,flag)
            matrix[i,j]=0
    return (matrix,0)

print("[Loading] model")
model = load_model('./digit_recognizer/model.h5')

# Reading the image
img_path = './sample_sudoku_images/sudoku1.png'
#img_path = './sample_sudoku_images/sudoku2.png'
#img_path = './sample_sudoku_images/sudoku3.jpg'
img = cv2.imread(img_path)
cv2.imshow('Original',img)

# Applying thresholding on sudoku grid
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray,(7,7),3)
thresh = cv2.adaptiveThreshold(blurred,255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
cv2.imshow('Thresholded',thresh)

# Finding contours on image
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(contours, key=cv2.contourArea, reverse=True)

# Get the largest perimeter contour i.e exact sudoku grid
peri = cv2.arcLength(cnts[0], True)
approx = cv2.approxPolyDP(cnts[0], 0.02 * peri, True)
cntr = cv2.drawContours(img.copy(),[approx],0,(0,255,0),2)
cv2.imshow('Sudoku_contours',cntr)

# Change the perspective of image to only sudoku grid
img = four_point_transform(img, approx.reshape(4, 2))
gray = four_point_transform(gray, approx.reshape(4, 2))
cv2.imshow('Sudoku_Perspective',gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


(h,w) = gray.shape[:2]
height = h//9
width = w//9

matrix = np.zeros((9,9),dtype='int')
# Get each digit in image
for i in range(9):
    cv2.imshow('Original',gray)
    for j in range(9):
        x = j*width
        y = i*height 
        roi = gray[y:y+height,x:x+width]

        blurred = cv2.GaussianBlur(roi,(15,15),3)
        
        thresh = cv2.adaptiveThreshold(blurred,255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
        thresh_no_border = clear_border(thresh)

        input_digit = cv2.resize(thresh_no_border,(28,28))
        component = connected(input_digit)
        if component is None:
            continue
	
        cv2.imshow('Thresholded_grid',np.hstack([roi,blurred,thresh,thresh_no_border]))
        cv2.imshow('Connected',component)
        

        roi = component.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        pred = model.predict(roi).argmax(axis=1)[0]

        print(i,j,pred)
        matrix[i,j] = pred
        cv2.waitKey(0)
    cv2.destroyAllWindows()

print("Sudoku :")
print(matrix)
cv2.imshow('Sudoku_grid',gray)
(sol,flag) = solve(matrix.copy(),0,0)
print("Sudoku_Solved : ")
print(sol)

# Fill the sudoku grid with answer
for i in range(9):
    for j in range(9):
        if(matrix[i][j]!=0):
            continue
        x = j*width
        y = i*height
        cv2.putText(img,str(sol[i][j]),(x+15,y+height-10),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
cv2.imshow('Sudoku_solved',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
