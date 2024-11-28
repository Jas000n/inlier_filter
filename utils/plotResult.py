import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(train_metrics_list, test_metrics_list, save_path):
    # 转换为 NumPy 数组
    train_metrics = np.array(train_metrics_list)
    test_metrics = np.array(test_metrics_list)

    epochs = np.arange(1, train_metrics.shape[0] + 1)

    # 创建绘图
    plt.figure(figsize=(18, 5))  # 调整画布大小以适应新的子图

    # 总损失图
    plt.subplot(1, 4, 1)
    plt.plot(epochs, train_metrics[:, 0], label='Train Total Loss')
    plt.plot(epochs, test_metrics[:, 0], label='Test Total Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Contrastive Loss
    plt.subplot(1, 4, 2)
    plt.plot(epochs, train_metrics[:, 1], label='Train Contrastive Loss')
    plt.plot(epochs, test_metrics[:, 1], label='Test Contrastive Loss')
    plt.title('Contrastive Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Classification Loss
    plt.subplot(1, 4, 3)
    plt.plot(epochs, train_metrics[:, 2], label='Train Classification Loss')
    plt.plot(epochs, test_metrics[:, 2], label='Test Classification Loss')
    plt.title('Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy 图
    plt.subplot(1, 4, 4)
    plt.plot(epochs, train_metrics[:, 3], label='Train Accuracy')
    plt.plot(epochs, test_metrics[:, 3], label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_classification_accuracy(train_metrics_list, test_metrics_list, save_path):
    # Convert lists to NumPy arrays
    train_metrics = np.array(train_metrics_list)
    test_metrics = np.array(test_metrics_list)
    
    # Define epochs based on the number of entries
    epochs = np.arange(1, train_metrics.shape[0] + 1)
    
    # Setup the figure
    plt.figure(figsize=(10, 5))  # Smaller figure for fewer plots

    # Classification Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_metrics[:, 0], label='Train Classification Loss')
    plt.plot(epochs, test_metrics[:, 0], label='Test Classification Loss')
    plt.title('Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_metrics[:, 1], label='Train Accuracy')
    plt.plot(epochs, test_metrics[:, 1], label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    train_metrics_list = [[0.4, 0.2, 0.2, 0.85], [0.35, 0.15, 0.2, 0.90], [0.3, 0.1, 0.2, 0.95]]
    test_metrics_list = [[0.45, 0.25, 0.2, 0.80], [0.4, 0.2, 0.2, 0.85], [0.35, 0.15, 0.2, 0.90]]

    save_path = './metrics_plot.png'
    plot_metrics(train_metrics_list, test_metrics_list, save_path)
