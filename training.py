import numpy as np
import tensorflow as tf
import datetime


def print_log(log):
    print(
        f"\033[96m[{datetime.datetime.now().strftime('%x %X')}][-][LOG]      \033[00m{log}")


def print_info(info):
    print(
        f"\033[93m[{datetime.datetime.now().strftime('%x %X')}][*][INFO]     {info}\033[00m")


def print_error(err):
    print(
        f"\033[91m[{datetime.datetime.now().strftime('%x %X')}][!][ERROR]    {err}\033[00m")


def print_success(info):
    print(
        f"\033[92m[{datetime.datetime.now().strftime('%x %X')}][+][SUCCESS]  {info}\033[00m")


def print_main(main, start=True):
    if start:
        sym = "^"
    else:
        sym = "$"
    print(
        f"\033[95m[{datetime.datetime.now().strftime('%x %X')}][{sym}][INFO]     {main}\033[00m")


print_main("Starting Programme")


# Load data
print_log("Loading data started")
# data_file = "data/fer2013.csv"
data_file = "data/temp.csv"
with open(data_file) as f:
    content = f.readlines()

lines = np.array(content)

num_of_instances = lines.size
print_success("Data loading done.")
print_info("Number of  total instances: " + str(num_of_instances))


# # Starting training
print_log("Starting Pre processing for data")
print_log("[Pre-Processing] Staring")
x_train, y_train, x_test, y_test = [], [], [], []
num_classes = 7

print_log("[Pre-Processing] Splitting data into training and test samples")
# # transfer train and test set data
for i in range(1, num_of_instances):
    try:
        emotion, img, usage = lines[i].split(",")

        val = img.split(" ")

        pixels = np.array(val, 'float32')

        emotion = tf.keras.utils.to_categorical(emotion, num_classes)

        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)
    except NameError:
        print_error(f"Something went wrong while splitting data: {NameError}")

# data transformation for train and test sets
x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')


print_log("[Pre-Processing] Normalizing pixel values from [0,255] to [0,1]")
x_train /= 255  # normalize inputs between [0, 1]
x_test /= 255


print_log("[Pre-Processing] Reshaping data to 48x48")
x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')

print_info(f'Train samples: {x_train.shape[0]}')
print_info(f'Test samples: {x_test.shape[0]}')
print_success("[Pre-Processing] Done")


print_main("Ending Programme", False)
