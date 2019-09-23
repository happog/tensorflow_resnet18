import tensorflow as tf
import numpy as np


class Model():
    def __init__(self):
        super().__init__()
        self.use_shortcut = True
        self.conv_0 = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same')

        self.block_1 = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization()
        ])
        self.shortcut_1 = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same'),
            tf.keras.layers.BatchNormalization()
        ])

        self.block_2 = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization()
        ])
        self.shortcut_2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2), padding='same'),
            tf.keras.layers.BatchNormalization()
        ])

        self.block_3 = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization()
        ])
        self.shortcut_3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(2, 2), padding='same'),
            tf.keras.layers.BatchNormalization()
        ])

        self.block_4 = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization()
        ])
        self.shortcut_4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(2, 2), padding='same'),
            tf.keras.layers.BatchNormalization()
        ])

        self.avg_pool = tf.keras.layers.AveragePooling2D((7, 7))
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(11)

    def forward(self, inputs):
        out = self.conv_0(inputs)
        tmp = out
        out = self.block_1(out)
        out += self.shortcut_1(tmp)

        tmp = out
        out = self.block_2(out)
        out += self.shortcut_2(tmp)

        tmp = out
        out = self.block_3(out)
        out += self.shortcut_3(tmp)

        tmp = out
        out = self.block_4(out)
        out += self.shortcut_4(tmp)

        out = self.avg_pool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


if __name__ == "__main__":
    img_size = 224
    img = tf.Variable(np.random.randn(1, img_size, img_size, 3).astype(np.float32))
    model = Model()
    res = model.forward(img)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print("Res : ", sess.run(res))
        print("Res shape: ", sess.run(tf.shape(res)))

