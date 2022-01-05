from PIL import Image
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import mnist

X_train =mnist.train_images()
y_train = mnist.train_labels()

X_test = mnist.test_images()
y_test = mnist.test_labels()

X_train = X_train.reshape((-1,28*28))
X_test = X_test.reshape((-1,28*28))

#we need to change the range of numbers from 0-255 to 0-1

X_train = (X_train/256)
X_test = (X_test/256)

classifier = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(64,64))
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
acc = confusion_matrix(y_test, predictions)
def accuracy(confusion_matrix):
    diagonal = confusion_matrix.trace()
    elements = confusion_matrix
    return diagonal/elements
print(accuracy(acc))

img = Image.open('handwirte.png')

data = list(img.getdata())

for i in range(len(data)):
    data[i] = 255-data[i]


digit = np.array(data)

p = classifier.predict(digit)
#will predict digit of handwritten image
print(p)

