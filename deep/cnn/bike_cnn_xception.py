import os
import sys
import numpy as np
from scipy.misc import toimage
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image


def plot_cifar10(X, y, result_dir):
    plt.figure()

    # 画像を描画
    nclasses = 10
    pos = 1
    for targetClass in range(nclasses):
        targetIdx = []
        # クラスclassIDの画像のインデックスリストを取得
        for i in range(len(y)):
            if y[i][0] == targetClass:
                targetIdx.append(i)

        # 各クラスからランダムに選んだ最初の10個の画像を描画
        np.random.shuffle(targetIdx)
        for idx in targetIdx[:10]:
            img = toimage(X[idx])
            plt.subplot(10, 10, pos)
            plt.imshow(img)
            plt.axis('off')
            pos += 1

    plt.savefig(os.path.join(result_dir, 'plot.png'))


def save_history(history, result_file):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(result_file, "w") as fp:
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(nb_epoch):
            fp.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss[i], acc[i], val_loss[i], val_acc[i]))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("usage: python cifar10.py [nb_epoch] [use_data_augmentation (True or False)] [result_dir]")
        exit(1)

    nb_epoch = int(sys.argv[1])
    data_augmentation = True if sys.argv[2] == "True" else False
    result_dir = sys.argv[3]
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    print("nb_epoch:", nb_epoch)
    print("data_augmentation:", data_augmentation)
    print("result_dir:", result_dir)

    batch_size = 128
    nb_classes = 2

    # 入力画像の次元
    img_rows, img_cols = 32, 32

    # チャネル数（RGBなので3）
    img_channels = 3
    
    label_list = []
    image_list = []

    for dir in os.listdir("bike_data/data/goonet/sr400/"):
        if dir == ".DS_Store":
            continue

        dir1 = "bike_data/data/goonet/sr400/" + dir 
        label = 0

        if dir == "right":    # appleはラベル0
            label = 0
        elif dir == "right_distotion": # orangeはラベル1
            label = 1
        
        for file in os.listdir(dir1):
            if dir == "right" or dir == "right_distotion":
                # 配列label_listに正解ラベルを追加(りんご:0 オレンジ:1)
                label_list.append(label)
                filepath = dir1 + "/" + file
                # 画像を25x25pixelに変換し、1要素が[R,G,B]3要素を含む配列の25x25の２次元配列として読み込む。
                # [R,G,B]はそれぞれが0-255の配列。
                image = np.array(Image.open(filepath).resize((71, 71)))
                # 配列を変換し、[[Redの配列],[Greenの配列],[Blueの配列]] のような形にする。
                image = image.transpose(2, 0, 1)
                # さらにフラットな1次元配列に変換。最初の1/3はRed、次がGreenの、最後がBlueの要素がフラットに並ぶ。
                #image = image.reshape(1, image.shape[0] * image.shape[1] * image.shape[2]).astype("float32")[0]
                image = np.transpose(image,(1,2,0))
                print(image.shape)
                # 出来上がった配列をimage_listに追加。
                image_list.append(image / 255.)
        
    # kerasに渡すためにnumpy配列に変換。
    image_list = np.array(image_list)

    # ラベルの配列を1と0からなるラベル配列に変更
    # 0 -> [1,0], 1 -> [0,1] という感じ。
    Y = to_categorical(label_list)

    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(image_list, Y, test_size=0.10, random_state=10)

    # CNNを構築
    from keras.applications.vgg16 import VGG16
    from keras.applications.xception import Xception
    from keras.models import Model
    base_model = Xception(include_top=False, weights=None, input_tensor=None, input_shape=(71,71,3),classes=2)
    x = base_model.output
    x = Flatten()(x)
    y = Dense(2,activation='softmax')(x)
    model = Model(inputs=base_model.input,outputs=y)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # モデルのサマリを表示
    model.summary()
    plot_model(model, show_shapes=True, to_file=os.path.join(result_dir, 'model.png'))
    
    # 訓練
    if not data_augmentation:
        print('Not using data augmentation')
        history = model.fit(X_train, y_train,
                            batch_size=batch_size,
                            nb_epoch=nb_epoch,
                            verbose=1,
                            validation_data=(X_test,y_test),
                            shuffle=True)
    else:
        print('Using real-time data augmentation')

        # 訓練データを生成するジェネレータ
        train_datagen = ImageDataGenerator(zca_whitening=True, width_shift_range=0.1, height_shift_range=0.1)
        train_datagen.fit(X_train)
        train_generator = train_datagen.flow(X_train, Y_train, batch_size=batch_size)

        # テストデータを生成するジェネレータ
        # 画像のランダムシフトは必要ない？
        test_datagen = ImageDataGenerator(zca_whitening=True)
        test_datagen.fit(X_test)
        test_generator = test_datagen.flow(X_test, Y_test)

        # ジェネレータから生成される画像を使って学習
        # 本来は好ましくないがテストデータをバリデーションデータとして使う
        # validation_dataにジェネレータを使うときはnb_val_samplesを指定する必要あり
        # TODO: 毎エポックで生成するのは無駄か？
        history = model.fit_generator(train_generator,
                                      samples_per_epoch=X_train.shape[0],
                                      nb_epoch=nb_epoch,
                                      validation_data=test_generator,
                                      nb_val_samples=X_test.shape[0])

    # 学習したモデルと重みと履歴の保存
    model_json = model.to_json()
    with open(os.path.join(result_dir, 'model.json'), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(result_dir, 'model.h5'))
    save_history(history, os.path.join(result_dir, 'history.txt'))

    # モデルの評価
    # 学習は白色化した画像を使ったので評価でも白色化したデータで評価する

    #if not data_augmentation:
    #    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    #else:
    #    loss, acc = model.evaluate_generator(test_generator, val_samples=X_test.shape[0])

    #print('Test loss:', loss)
    #print('Test acc:', acc)
