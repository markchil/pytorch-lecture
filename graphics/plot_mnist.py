import matplotlib.pyplot as plt
from torchvision.datasets import MNIST

if __name__ == '__main__':
    plt.ion()
    plt.close('all')

    train_val_dataset = MNIST('./data', train=True, download=True)

    num_panel = 4
    f, a = plt.subplots(1, num_panel)
    for i_samp in range(num_panel):
        image, label = train_val_dataset[i_samp]
        a[i_samp].imshow(image, cmap='Greys')
        a[i_samp].set_title(f'$y={label}$')
        a[i_samp].xaxis.set_visible(False)
        a[i_samp].yaxis.set_visible(False)

    f.savefig('mnist.png', bbox_inches='tight')
