import torch
import torch.nn as nn
from torch.optim import Adam
import tqdm
import random
import fire
from dataclasses import dataclass
from typing import Literal

# Import model-specific modules
from local_transformer import get_local_transformer, LocalTransformerConfig
from mac import MACTransformer, MACConfig
from data.shakespeare.dataloader import get_dataloaders, decode_tokens


@dataclass
class TrainingConfig:
    """Configuration for training transformer models."""
    learning_rate: float = 1e-4
    num_batches: int = 10000
    batch_size: int = 8
    seq_len: int = 1024
    gradient_accumulate_every: int = 4
    validate_every: int = 100
    generate_every: int = 250
    generate_length: int = 200


def get_model(model_type: str, seq_len: int):
    """Get the specified model type with appropriate configuration."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_type == "local":
        model_config = LocalTransformerConfig(
            max_seq_len=seq_len,
            device=device
        )
        model = get_local_transformer(model_config)
    elif model_type == "mac":
        model_config = MACConfig(
            d_model=512,
            seq_len=seq_len,
            num_tokens=256
        )
        model = MACTransformer(model_config)
        if device == 'cuda':
            model = model.cuda()
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'local' or 'mac'.")
    
    return model


def train(model_type: Literal["local", "mac"] = "local"):
    """
    Train a transformer model.
    
    Args:
        model_type: Type of model to train ("local" or "mac")
    """
    print(f"Training {model_type} model...")
    
    # Initialize training configuration
    train_config = TrainingConfig()
    
    # Initialize model
    model = get_model(model_type, train_config.seq_len)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        batch_size=train_config.batch_size,
        seq_len=train_config.seq_len
    )
    
    def infinite(dataloader):
        while True:
            for batch in dataloader:
                yield batch
    
    train_loader = infinite(train_loader)
    val_loader = infinite(val_loader)
    
    # Get validation dataset for text generation
    _, val_dataset_loader = get_dataloaders(
        batch_size=1, 
        random_sampling=False,
        seq_len=train_config.seq_len
    )
    val_dataset = []
    for batch in val_dataset_loader:
        val_dataset.extend(batch)
        if len(val_dataset) >= 100:  # Get enough samples for generation
            break
    
    # optimizer
    optim = Adam(model.parameters(), lr=train_config.learning_rate)
    
    # training loop
    for i in tqdm.tqdm(range(train_config.num_batches), mininterval=10., desc=f'training {model_type}'):
        model.train()
        
        for __ in range(train_config.gradient_accumulate_every):
            batch = next(train_loader)
            input_seq = batch[:, :-1].long()  # Shape: (batch_size, seq_len)
            target_seq = batch[:, 1:].long()  # Shape: (batch_size, seq_len)
            
            logits = model(input_seq)  # Shape: (batch_size, seq_len, num_tokens)
            loss = nn.CrossEntropyLoss()(logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1))
            loss.backward()
        
        print(f'training loss: {loss.item()}')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()
        
        if i % train_config.validate_every == 0:
            model.eval()
            with torch.no_grad():
                val_batch = next(val_loader)
                val_input = val_batch[:, :-1].long()
                val_target = val_batch[:, 1:].long()
                val_logits = model(val_input)
                val_loss = nn.CrossEntropyLoss()(val_logits.reshape(-1, val_logits.size(-1)), val_target.reshape(-1))
                print(f'validation loss: {val_loss.item()}')
        
        if i % train_config.generate_every == 0:
            model.eval()
            with torch.no_grad():
                # Take a random sequence from validation data as seed
                seed_seq = random.choice(val_dataset)[:-1].long()
                prime = decode_tokens(seed_seq.cpu().tolist())
                print(f'\nSeed text: {prime}')
                print('*' * 100)
                
                # Generate new text (assuming the model has a generate method)
                if hasattr(model, 'generate'):
                    sample = model.generate(seed_seq[None, ...], train_config.generate_length)
                    output_str = decode_tokens(sample[0].cpu().tolist())
                    print(f'Generated text: {output_str}')
                else:
                    print("Model doesn't have generate method - skipping text generation")
                print('*' * 100)
    
    print(f"Training {model_type} model completed!")


if __name__ == "__main__":
    fire.Fire(train)
