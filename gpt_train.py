import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm

from gpt_model import GPT

# Hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
eval_iters = 200
n_embd = 384
n_layers = 6
n_heads = 6
dropout = 0.2
# --------------------

torch.manual_seed(1337)

# -------------------- Download the dataset --------------------
 # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r') as f:
    text = f.read()


# -------------------- Tokenize the dataset --------------------
# Get all unique characters in the dataset
unique_chars = sorted(list(set(text)))
vocab_size = len(unique_chars)
# print(''.join(unique_chars))
# print(vocab_size)

# Create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(unique_chars) }
itos = { i:ch for i,ch in enumerate(unique_chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
# print(encode("hii there"))
# print(decode(encode("hii there")))

# Create a tensor of the dataset
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000])


# -------------------- Train-Validation Split --------------------
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
# print(len(train_data)/len(data), len(val_data)/len(data))


# -------------------- Create a DataLoader --------------------
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


# -------------------- Estimate the loss --------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# -------------------- Train the model --------------------
model = GPT(vocab_size, block_size, n_embd, n_layers, n_heads, dropout)
model = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in tqdm.tqdm(range(max_iters)):
    # every eval_interval, check the loss on the validation set
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # sample a batch of data
    X, Y = get_batch('train')
    
    # evaluate the loss
    logits, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# -------------------- Generate from the model --------------------
context = torch.zeros((1, 1), dtype=torch.long, device=device)
#print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
open('more.txt', 'w').write(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))

# -------------------- Save the model --------------------
torch.save(model.state_dict(), 'gpt.pth')