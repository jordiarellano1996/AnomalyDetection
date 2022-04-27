import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from asteroidlib.plot_factory import *
from skimage.feature.peak import peak_local_max
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import color
from sklearn.neighbors import KernelDensity


def plot_anomaly_position(img_data, img_predicted):
    img_data = color.rgb2gray(img_data)
    img_pred = color.rgb2gray(img_predicted)
    squared_diff_img = (img_data - img_pred) ** 2
    coordinates = peak_local_max(squared_diff_img, min_distance=30, num_peaks=5, threshold_rel=0.8)

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(img_data, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('Image Original')

    ax[1].imshow(img_pred, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('Image Predicted')

    ax[2].imshow(img_pred, cmap=plt.cm.gray)
    ax[2].autoscale(False)
    # ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    width = 50
    for i in range(len(coordinates)):
        ax[2].add_patch(
            Rectangle((coordinates[i, 1] - (width // 2), coordinates[i, 0] - (width // 2)),
                      width, width,
                      linewidth=1,
                      edgecolor='r',
                      facecolor='none'))
    ax[2].axis('off')
    ax[2].set_title('Peak local max')

    fig.tight_layout()
    plt.show()


def check_anomaly(image, model_in, encoder_model_in, kde_in,
                  density_threshold=1091):
    img = image[np.newaxis, :, :, :]
    encoder_img = encoder_model_in.predict(img)
    flatten_e_image = encoder_img.reshape(encoder_img.shape[0],
                                          encoder_img.shape[1] *
                                          encoder_img.shape[2] *
                                          encoder_img.shape[3])

    density = kde_in.score_samples(flatten_e_image)[0]
    reconstruction = model_in.predict(img)
    reconstruction_err = model_in.evaluate(reconstruction, img, batch_size=1)[0]
    print(density, reconstruction_err)

    if density < density_threshold:
        print("anomaly")
        return True
    else:
        return False


def calc_density_and_recon_error(image_generator, model_in, encoder_model_in, kde_in):
    # To set a threshold, first we need to find which are the values in normal images.
    data_batch = []
    img_num = 0
    while img_num <= image_generator.batch_index:
        data = image_generator.next()
        data_batch.extend(data[0])
        img_num += 1
    data_batch = np.array(data_batch)
    encoded_predict_imgaes = encoder_model_in.predict(image_generator)
    flatten_ep_images = encoded_predict_imgaes.reshape(encoded_predict_imgaes.shape[0],
                                                       encoded_predict_imgaes.shape[1] *
                                                       encoded_predict_imgaes.shape[2] *
                                                       encoded_predict_imgaes.shape[3])
    # Compute the log-likelihood of each sample under the model.
    density_arr = kde_in.score_samples(flatten_ep_images)
    reconstruction = model_in.predict(image_generator)
    reconstruction_err = model_in.evaluate(reconstruction, data_batch)
    del data_batch
    average_density = np.mean(density_arr)
    stdev_density = np.std(density_arr)

    return average_density, stdev_density, reconstruction_err, density_arr


if __name__ == "__main__":
    MODEL_PATH = "/home/titoare/Documents/IFAE/asteroid/logs/log_256_640_24_1651001547"
    IMAGE_SIZE_Y = 256
    IMAGE_SIZE_X = 640
    IMAGE_BATCH_SIZE = 24
    datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = datagen.flow_from_directory(
        '/home/titoare/Documents/IFAE/asteroid/capsule_data_cropped/train/',
        target_size=(IMAGE_SIZE_Y, IMAGE_SIZE_X),
        batch_size=IMAGE_BATCH_SIZE,
        class_mode='input',
        save_format='png',
    )

    validation_generator = datagen.flow_from_directory(
        '/home/titoare/Documents/IFAE/asteroid/capsule_data_cropped/test/',
        target_size=(IMAGE_SIZE_Y, IMAGE_SIZE_X),
        batch_size=IMAGE_BATCH_SIZE,
        class_mode='input',
        save_format='png',
    )

    anomaly_generator = datagen.flow_from_directory(
        '/home/titoare/Documents/IFAE/asteroid/capsule_data_cropped/test_bad/',
        target_size=(IMAGE_SIZE_Y, IMAGE_SIZE_X),
        batch_size=IMAGE_BATCH_SIZE,
        class_mode='input',
        save_format='png',
    )

    # Model creation
    model = load_model(MODEL_PATH + "/complete_model")

    # Examine the reconstruction error between train data, validation data and anomaly data
    train_error = model.evaluate(train_generator)
    validation_error = model.evaluate(validation_generator)
    anomaly_error = model.evaluate(anomaly_generator)
    print(f"train_error--> [mse, mae] --> {train_error}")
    print(f"validation_error--> [mse, mae] --> {validation_error}")
    print(f"anomaly_error--> [mse, mae] --> {anomaly_error}")

    # Model prediction
    train_data = train_generator[0][0]
    train_predict = model.predict(train_generator[0][0])
    validation_data = validation_generator[0][0]
    validation_predict = model.predict(validation_generator[0][0])
    anomaly_data = anomaly_generator[0][0]
    anomaly_predict = model.predict(anomaly_generator[0][0])

    # Encoder model
    encoder_model = load_model(MODEL_PATH + "/encoder_complete_model")

    # Calculate KDE using sklearn
    encoded_predicted_images = encoder_model.predict(train_generator)
    flatten_ep_images = encoded_predicted_images.reshape(encoded_predicted_images.shape[0],
                                                         encoded_predicted_images.shape[1] *
                                                         encoded_predicted_images.shape[2] *
                                                         encoded_predicted_images.shape[3])

    kde = KernelDensity(kernel="gaussian", bandwidth=0.2).fit(flatten_ep_images)

    train_dr_error = calc_density_and_recon_error(train_generator, model, encoder_model, kde)
    validation_dr_error = calc_density_and_recon_error(validation_generator, model, encoder_model, kde)
    anomaly_dr_error = calc_density_and_recon_error(anomaly_generator, model, encoder_model, kde)

    # If the image presents anomaly plot where.
    for i in range(len(anomaly_data)):
        anomaly_flag = check_anomaly(anomaly_data[i], model, encoder_model, kde, density_threshold=1091)
        if anomaly_flag:
            plot_anomaly_position(anomaly_data[i], anomaly_predict[i])
