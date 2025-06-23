"""Repeatable code parts concerning data loading."""


import logging
import datasets

def load_data(dataset, cache="~/data"):
    """Return a dataloader with given dataset and augmentation, normalize data?."""
    if dataset == 'BC2GM':
        dataset = datasets.load_dataset('spyysalo/bc2gm_corpus', cache_dir=cache)
    elif dataset == 'BC4CHEMD':
        dataset = datasets.load_dataset(
            'chintagunta85/bc4chemd', cache_dir=cache
        )
    elif dataset == 'CONLL-2003':
        dataset = datasets.load_dataset(
            'eriktks/conll2003', cache_dir=cache
        )
    else:
        raise ValueError(f"Unknown dataset {dataset}.")

    return dataset

def num_classes(dataset):
    """Return the number of classes for a given dataset."""
    if dataset == 'BC2GM':
        return 3
    elif dataset == 'BC4CHEMD':
        return 5
    elif dataset == 'CONLL-2003':
        return 9
    else:
        raise ValueError(f"Unknown dataset {dataset}.")


def create_partitions(
    dataset,
    num_partitions: int,
    seed: int = 42,
):
    """
    Create partitioned versions of a source dataset with shuffling.

    Args:
        dataset (Dataset): The source dataset to partition.
        num_partitions (int): The number of partitions to create.
        seed (int): Random seed for shuffling.

    Returns:
        list: A list of datasets, each representing a partition.
    """
    # Shuffle the dataset with a random seed
    shuffled_dataset = dataset.shuffle(seed=seed)

    # Calculate the size of each partition
    partition_size = len(shuffled_dataset) // num_partitions

    partitions = []

    for i in range(num_partitions):
        # Define the start and end indices for the partition
        start_idx = i * partition_size
        end_idx = (
            (i + 1) * partition_size
            if i != num_partitions - 1
            else len(shuffled_dataset)
        )

        # Create the partition dataset
        partition = shuffled_dataset.select(range(start_idx, end_idx))
        partitions.append(partition)

    return partitions

def get_partition(pid, total_clients, dataset):
    """
    Get the partition of the dataset for the client.

    Args:
        pid (int): The partition ID.
        total_clients (int): The total number of clients.
        dataset (Dataset): The source dataset.

    Returns:
        Dataset: The partition of the dataset for the client.
    """
    client_idx = pid - 1

    train_partitions = create_partitions(dataset["train"], total_clients)
    test_partitions = create_partitions(dataset["test"], total_clients)
    validation_partitions = create_partitions(
        dataset["validation"], total_clients
    )

    logging.info(
        f"Number of training examples for client {pid}:"
        f" {train_partitions[client_idx].num_rows}"
    )
    logging.info(
        f"Number of test examples for client {pid}:"
        f" {test_partitions[client_idx].num_rows}"
    )
    logging.info(
        f"Number of validation examples for client {pid}:"
        f" {validation_partitions[client_idx].num_rows}"
    )

    return {
        "train": train_partitions[client_idx],
        "test": test_partitions[client_idx],
        "validation": validation_partitions[client_idx],
    }


if __name__ == "__main__":
    dataset = load_data('BC2GM')
    print(dataset)
    partition = get_partition(1, 2, dataset)
    print(partition)
    print(num_classes('BC2GM'))

    # Print one example from the Dataset
    print(dataset['train'][0])
    # Print the number of classes
    print(num_classes('BC2GM'))


