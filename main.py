from flask import Flask, render_template, request
import cv2, numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

app = Flask(__name__)
app.debug = True


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/task', methods=['POST'])
def task():
    if request.method == 'POST':
        with_mask = np.load('with mask.npy')
        without_mask = np.load('without mask.npy')

        with_mask.shape
        without_mask.shape
        with_mask = with_mask.reshape(200, 50 * 50 * 3)
        without_mask = without_mask.reshape(200, 50 * 50 * 3)
        with_mask.shape
        x = np.r_[with_mask, without_mask]
        x.shape
        labels = np.zeros(x.shape[0])
        labels[200:] = 1.0
        names = {0: 'mask', 1: 'no mask'}
        x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.25)
        x_train.shape
        pca = PCA(n_components=3)
        x_train = pca.fit_transform(x_train)
        x_train[0]
        x_train.shape
        x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.20)
        svm = SVC()
        svm.fit(x_train, y_train)
        y_pred = svm.predict(x_test)
        accuracy_score(y_test, y_pred)
        haar_data = cv2.CascadeClassifier('C:/Users/Shraddha/PycharmProjects/pythonProject')

        capture = cv2.VideoCapture(0)
        data = []
        font = cv2.FONT_HERSHEY_COMPLEX
        while True:
            flag, img = capture.read()
            if flag:
                faces = haar_data.detectMultiScale(img)
                for x, y, w, h in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 4)
                    face = img[y:y + h, x:x + w, :]
                    face = cv2.resize(face, (50, 50))
                    face = face.reshape(1, -1)
                    # face = pca.transform(face)
                    pred = svm.predict(face)
                    n = names[int(pred)]
                    cv2.putText(img, n, (x, y), font, 1, (244, 250, 250), 2)
                    print(n)
                cv2.imshow('result', img)

                if cv2.waitKey(2) == 27:
                    break

        capture.release()
        cv2.destroyAllWindows()

        return render_template('output.html', result=result1)


if __name__ == '__main__':
    app.run()
