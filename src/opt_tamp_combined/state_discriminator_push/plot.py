import matplotlib.pyplot as plt
from utils import load_list


def main():
    # Get data
    losses = load_list('losses')
    accuracies = load_list('accur')    

    # Plotting
    # plotting the loss
    plt.plot(losses)
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('./loss.png')
    plt.show()

    # plotting the accuracy
    plt.plot(accuracies)
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('./accuracy.png')
    plt.show()



if __name__ == '__main__':
    main()