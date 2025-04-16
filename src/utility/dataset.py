import logging

from datasets import Dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

# Create a logger specific to this module
logger = logging.getLogger(__name__)

def train_valid_test_split(dataset:Dataset, seed:int=None):
    """
    Split dataset into train, valid and test set
    
    Args:
        dataset (Dataset): loaded huggingface dataset
        seed (int): seed value for train_test_split()

    Returns
        train_set (Dataset)
        validation_set (Dataset)
        test_set (Dataset)
    """
    # split dataset into train and test
    train_test_split = dataset.train_test_split(test_size=0.2, seed=seed)
    
    # split test set into valid and test
    validation_test_split = train_test_split['test'].train_test_split(test_size=0.5, seed=seed)

    # New splits
    train_set = train_test_split['train']
    validation_set = validation_test_split['train']
    test_set = validation_test_split['test']

    return train_set, validation_set, test_set



class CustomFederatedDataset(FederatedDataset):
    def __init__(self, dataset, num_clients):
        super().__init__()
        self.dataset = dataset
        self.num_clients = num_clients
        
        # Create IID partitioner with the specified number of partitions (one per client)
        self.partitioner = IidPartitioner(num_partitions=num_clients)

    def get_partitions(self):
        # Use the partitioner to create partitions for each client
        partitions = self.partitioner(self.dataset)
        
        # Collect the partitions for each client
        client_partitions = [self.dataset.select(indices) for indices in partitions]
        
        return client_partitions
