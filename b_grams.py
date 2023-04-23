import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyper parameters
config = {
    "batch_size": 32,
    "block_size": 8,
    "max_iters": 3000,
    "eval_interval": 300,
    "learning_rate": 1e-2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "eval_iters": 200,
}

torch.manual_seed(1337)


def load_data(file_path):
    """Load text data from a file.

    Args:
        file_path (str): Path to the input file.

    Returns:
        str: The content of the input file as a string.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def preprocess_data(text):
    """Preprocess the input text for training.

    Args:
        text (str): Input text.

    Returns:
        tuple: A tuple containing the vocabulary size, train_data, val_data, encode function, and decode function.
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return vocab_size, train_data, val_data, encode, decode


def get_batch(data, batch_size, block_size, device):
    """Get a batch of data of inputs x and targets y.

    Args:
        data (torch.Tensor): The data tensor.
        batch_size (int): The number of independent sequences to process in parallel.
        block_size (int): The maximum context length for predictions.
        device (str): The device to use for computation ('cuda' or 'cpu').

    Returns:
        tuple: A tuple containing the input tensor x and the target tensor y.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def create_dataloader(data, batch_size, block_size, device):
    """Create a dataloader for the given data.

    Args:
        data (torch.Tensor): The data tensor.
        batch_size (int): The number of independent sequences to process in parallel.
        block_size (int): The maximum context length for predictions.
        device (str): The device to use for computation ('cuda' or 'cpu').

    Returns:
        generator: A generator that yields new batches on each iteration.
    """

    def data_generator():
        while True:
            x, y = get_batch(data, batch_size, block_size, device)
            yield x, y

    return data_generator()


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_iters, device):
    """Estimate the loss for the model on train and validation sets.

    Args:
        model (BigramLanguageModel): The model to evaluate.
        train_loader (function): A function that returns train data batches.
        val_loader (function): A function that returns validation data batches.
        eval_iters (int): The number of iterations to estimate the loss.
        device (str): The device to use for computation ('cuda' or 'cpu').

    Returns:
        dict: A dictionary containing the average loss for train and validation sets.
    """
    out = {}
    model.eval()
    for split, loader in [("train", train_loader), ("val", val_loader)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = next(loader)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):
    """A simple bigram language model."""

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """Perform a forward pass through the model.

        Args:
            idx (torch.Tensor): A tensor of input indices.
            targets (torch.Tensor, optional): A tensor of target indices.

        Returns:
            tuple: A tuple containing the logits tensor and the loss value.
        """
        logits = self.token_embedding_table(idx)  # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generate new tokens based on the current context.

        Args:
            idx (torch.Tensor): A tensor of input indices representing the current context.
            max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
            torch.Tensor: A tensor containing the generated indices.
        """
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]  # becomes (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


def main():
    # Load data and preprocess
    file_path = "./data/input.txt"
    text = load_data(file_path)
    vocab_size, train_data, val_data, encode, decode = preprocess_data(text)

    # Set device
    device = config["device"]

    # Initialize model and move to device
    model = BigramLanguageModel(vocab_size)
    model.to(device)

    # Create dataloaders
    train_loader = create_dataloader(train_data, config["batch_size"], config["block_size"], device)
    val_loader = create_dataloader(val_data, config["batch_size"], config["block_size"], device)

    # Create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    # Training loop
    for step in range(config["max_iters"]):

        # Estimate loss on train and val sets
        if step % config["eval_interval"] == 0:
            losses = estimate_loss(model, train_loader, val_loader, config["eval_iters"], device)
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Sample a batch of data
        xb, yb = next(train_loader())

        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Generate 100 characters from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_text = decode(model.generate(context, max_new_tokens=100)[0].tolist())
    print(generated_text)


def main():
    # Load data and preprocess
    file_path = "./data/input.txt"
    text = load_data(file_path)
    vocab_size, train_data, val_data, encode, decode = preprocess_data(text)

    # Set device
    device = config["device"]

    # Initialize model and move to device
    model = BigramLanguageModel(vocab_size)
    model.to(device)

    # Create dataloaders
    train_loader = create_dataloader(train_data, config["batch_size"], config["block_size"], device)
    val_loader = create_dataloader(val_data, config["batch_size"], config["block_size"], device)

    # Create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    # Training loop
    for step in range(config["max_iters"]):

        # Estimate loss on train and val sets
        if step % config["eval_interval"] == 0:
            losses = estimate_loss(model, train_loader, val_loader, config["eval_iters"], device)
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Sample a batch of data
        xb, yb = next(train_loader)

        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Generate 100 characters from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_text = decode(model.generate(context, max_new_tokens=100)[0].tolist())
    print(generated_text)


if __name__ == "__main__":
    main()
