from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical



def train_and_save_model():
    # Load MNIST data 
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # 28 for width and height, 1 for grayscale, -1 means any number of samples
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0 
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    y_train = to_categorical(y_train, 10) #converts integer labels to one-hot encoded labels
    y_test = to_categorical(y_test, 10)

    # build model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model, .fit() shuffles the data by default
    model.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test))

    # Save model
    model.save("mnist_cnn_model.h5")

if __name__ == "__main__":
    train_and_save_model()
