import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

clf = svm.SVC(gamma=0.000001, C=100)

print(len(digits.data))

x,y = digits.data[:-12], digits.target[:-12]
clf.fit(x,y)

print ('prediction:', clf.predict(digits.data[-1].reshape(1, -1)))

plt.imshow(digits.images[-1],cmap=plt.cm.gray_r, interpolation = "nearest")
plt.show()
