import os
import requests
import torch
from torch.utils.data import Dataset, DataLoader

def decode_token(token):
    return str(chr(max(32, int(token))))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].float()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

class SequentialTextDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        start_idx = index * self.seq_len
        full_seq = self.data[start_idx: start_idx + self.seq_len + 1].float()
        return full_seq.cuda()

    def __len__(self):
        return (self.data.size(0) - 1) // self.seq_len

def get_dataloaders(seq_len=1024, batch_size=32, random_sampling=True):
    """
    Returns train and validation dataloaders for the tiny Shakespeare dataset.
    
    Args:
        seq_len (int): Sequence length for each sample
        batch_size (int): Batch size for dataloaders
        random_sampling (bool): If True, uses random sampling. If False, uses sequential sampling.
    
    Returns:
        tuple: (train_loader, val_loader) - both are cyclic dataloaders
    """
    # Download the tiny shakespeare dataset
    input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
    if not os.path.exists(input_file_path):
        print("Downloading tiny shakespeare dataset...")
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w", encoding="utf-8") as f:
            f.write(requests.get(data_url).text)
        print("Download complete!")

    # Load data directly from text file as bytes
    with open(input_file_path, "rb") as f:
        data_bytes = f.read()

    # Convert bytes to tensor
    data_tensor = torch.tensor(list(data_bytes), dtype=torch.uint8)

    # Simple 90/10 train/val split
    train_split = 0.9
    n = len(data_tensor)
    data_train = data_tensor[:int(n * train_split)]
    data_val = data_tensor[int(n * train_split):]

    print(f"Total characters: {n:,}")
    print(f"Train characters: {len(data_train):,} ({len(data_train)/n*100:.1f}%)")
    print(f"Val characters: {len(data_val):,} ({len(data_val)/n*100:.1f}%)")
    print(f"Vocabulary size (unique bytes): {len(set(data_bytes))}")

    # Choose dataset type based on sampling method
    dataset_class = TextSamplerDataset if random_sampling else SequentialTextDataset
    
    # Create datasets
    train_dataset = dataset_class(data_train, seq_len)
    val_dataset = dataset_class(data_val, seq_len)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=random_sampling)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=random_sampling)    
    
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Val dataset length: {len(val_dataset)}")
    
    return train_loader, val_loader

if __name__ == "__main__":
    seq_len = 64
    batch_size = 4
    random_sampling = False

    train_loader, val_loader = get_dataloaders(seq_len, batch_size, random_sampling)

    print("First batch from train_loader:")
    for i, batch in enumerate(train_loader):
        for j, sequence in enumerate(batch):
            print(f"Batch index: {j}")
            print(f"Encoded: {sequence}")
            print(f"Decoded: {decode_tokens(sequence.cpu().tolist())}")
            print()  # Empty line for readability
        break
