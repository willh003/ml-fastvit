import matplotlib.pyplot as plt

############ Data Vis #################

def save_mask(image, mask, path):

    plt.imshow(image) # I would add interpolation='none'
    plt.imshow(mask, cmap='jet', alpha=0.5) # interpolation='none'
    plt.savefig(path)
