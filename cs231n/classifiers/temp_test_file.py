
import numpy as np

W = np.random.randn(3073, 10) * 0.0001
# print(W,W.shape)
X = [1,2,3,4,5,6,7,8]
mask = np.random.choice(X,5,replace=True)
print(mask,np.random.choice(X,5,replace=True))
# y = np.ones((10,3))
learning_rates = np.arange(0, 5e-5,5e-6)
print(learning_rates)
regularization_strengths = [(1+i*0.1)*1e4 for i in range(-3,3)] + [(2+0.1*i)*1e4 for i in range(-3,3)]
print(regularization_strengths)



import image
import sys

img = image.open("1.jpg")
width = img.size[0]
height = img.size[1]
for w in range(width):
    for h in range(height):
        pixel = img.getpixel(w, h)
        print(pixel)

