from torch.utils.data import random_split, DataLoader

class data_module:
    def __init__(self, dataset, test, split=(80, 20), batch_size=32):
        self.dataset = dataset
        self.test_dataset = test
        self.train_size, self.val_size = split
        self.train_dl, self.val_dl, self.test_dl = None, None, None
        self.batch_size = batch_size

    def split_data(self, shuffle=True):
        if self.train_dl is not None and self.val_dl is not None and self.test_dl is not None:
            print("Data already split. Skipping.")
            return self.train_dl, self.val_dl, self.test_dl

        total_size = len(self.dataset)
        train_size = int(total_size * self.train_size / 100)
        val_size = total_size - train_size

        train, val = random_split(self.dataset, [train_size, val_size])

        # Use persistent_workers and prefetch_factor to speed up data loading, especially for larger datasets. Adjust num_workers based on your system's capabilities.
        # Prefetching can help keep the GPU fed with data, reducing idle time and improving training speed. Each worker will prefetch 2 batches by default, which can be adjusted based on the dataset size and system memory.
        #Persistent workers keep the worker processes alive across epochs, which can further reduce overhead when loading data in subsequent epochs.

        self.train_dl = DataLoader(train, batch_size=self.batch_size, shuffle=shuffle, num_workers=4, persistent_workers=True, prefetch_factor=2) 
        self.val_dl = DataLoader(val, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True, prefetch_factor=2)
        self.test_dl = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True, prefetch_factor=2)

        print(f"Data split into {train_size} training and {val_size} validation samples")
        return self.train_dl, self.val_dl, self.test_dl