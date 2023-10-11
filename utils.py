from torch.utils.data import DataLoader, Dataset
from client_ds import ClientDataset
import numpy as np
import matplotlib.pyplot as plt

def get_client_labels(dataset, user_groups, num_workers, num_classes, proportions=False):
    """
    Creates a List containing the set of all labels present in both train and validation sets for each client,
    optionally returns this list of present lables or a List of proportions of each class in the dataset
    Args:
        dataset: the complete dataset being used
        user_groups: dict of indices assigned to each client
        num_workers: how many sub processes to use for data loading
        num_classes: number of classes in the dataset
        proportions: boolean indicating if class proportions should be returned instead of client labels
    Returns: if proportions is False: List containing the set of all labels present in both train and validation sets
    of each client dataset, indexed by client number. If proportions is True: a list containing the proportion of each
    label of each client dataset, indexed by client number.
    """
    def get_labels(client_idxs):
        dataloader = DataLoader(ClientDataset(dataset, client_idxs), batch_size=len(dataset), shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        _, labels = zip(*[batch for batch in dataloader])

        if proportions:
            labels = np.asarray(labels[0])
            count_labels = labels.shape[0]
            count_client_labels = []
            for i in range(num_classes):
                count_client_labels.append(int(np.argwhere(labels == i).shape[0]))
            count_client_labels = np.array(count_client_labels)
            return np.unique(labels), count_client_labels / count_labels

        return labels[0].unique()

    if proportions:
        client_groups = user_groups.items()
        client_labels = []
        client_proportions = []
        for client in client_groups:
            unique_labels, label_proportions = get_labels(np.concatenate((client[1]['train'], client[1]['validation']), axis=0))
            client_labels.append(unique_labels)
            client_proportions.append(label_proportions)
        return client_proportions
    else:
        client_groups = user_groups.items()
        client_labels = [get_labels(np.concatenate((client[1]['train'], client[1]['validation']), axis=0))
                        for client in client_groups]
        return client_labels


def visualize_props(user_labels, user_idx):
    categories = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7','L8', 'L9', 'L10']
    values = user_labels[user_idx]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink']
    plt.figure(figsize=(4, 2))
    plt.barh(categories, values, color=colors)
    plt.xlabel('Proportion')
    plt.ylabel('Classes')

    plt.show()