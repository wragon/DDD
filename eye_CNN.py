# 모듈 선언
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# dataset
# 출처 : https://github.com/kairess/eye_blink_detector/tree/master/dataset
x_train = np.load('C:/Users/user/Desktop/Junyong/PycharmProjects/dataset/x_train.npy').astype(np.float32)
y_train = np.load('C:/Users/user/Desktop/Junyong/PycharmProjects/dataset/y_train.npy').astype(np.float32)
x_val = np.load('C:/Users/user/Desktop/Junyong/PycharmProjects/dataset/x_val.npy').astype(np.float32)
y_val = np.load('C:/Users/user/Desktop/Junyong/PycharmProjects/dataset/y_val.npy').astype(np.float32)

# Image Augmentation(옵션을 더 많이 줘서 개선)
train_argmentation = ImageDataGenerator(
    rescale=1./255,  # 이미지 0-1 사이의 값으로 변경
    rotation_range=10,
    zoom_range=0.2,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False
)

val_argmentation = ImageDataGenerator(
    rescale=1./255
)

augment_size = 50
train_generator = train_argmentation.flow(
    x=x_train, y=y_train,
    batch_size=augment_size,
    shuffle=True
)
val_generator = val_argmentation.flow(
    x=x_val, y=y_val,
    batch_size=augment_size,
    shuffle=False
)

# CNN 모델 생성(tensorflow로 개선)
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(input_shape=(26, 34, 1), kernel_size=(3,3), filters=32, strides=1, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, padding='same', strides=1, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=128, padding='same', strides=1, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

cnn_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
print(cnn_model.summary())

# 모델 학습
history = cnn_model.fit_generator(train_generator, epochs=25, validation_data=val_generator)

# 학습 결과 시각화
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], 'g-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
plt.xlabel('Epoch')
plt.legend()

cnn_model.save("cnn_model.h5")
y_pred = cnn_model.predict(x_val/255.)
y_pred_logical = (y_pred > 0.5).astype(np.int)

print ('test acc: %s' % accuracy_score(y_val, y_pred_logical))
cm = confusion_matrix(y_val, y_pred_logical)

plt.subplot(1, 3, 3)
sns.heatmap(cm, annot=True)
plt.show()