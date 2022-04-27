import matplotlib.pyplot as plt


def plot_image(image):
    fig, ax = plt.subplots()
    fig.figimage(image, resize=True)
    plt.show()


def compare_images(image1, iamge2):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image1)
    ax[1].imshow(iamge2)
    plt.show()
