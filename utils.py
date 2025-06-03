import torch
import torch.nn.functional as F

def train_one_epoch(model, device, sp, optimizer, data_loader):
    model.train()
    optimizer.zero_grad()
    
    # Loop over batches in data_loader
    total_loss = 0.0
    for x_batch, y_batch in data_loader:
        # Move the batch to the correct device (GPU/CPU)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # Forward pass
        output = model(x_batch)  # output shape: (batch_size, seq_len, vocab_size)

        # Flatten output and target tensors
        output = output.view(-1, output.size(-1))  # Flatten to (batch_size * seq_len, vocab_size)
        y_batch = y_batch.view(-1)  # Flatten to (batch_size * seq_len)

        # Calculate loss
        loss = F.cross_entropy(output, y_batch, ignore_index=sp.pad_id())  # Use padding token if needed
        
        # Backpropagation
        loss.backward()

        # Optimizer step
        optimizer.step()
        
        # Clear GPU memory (optional, but useful for large batches)
        torch.cuda.empty_cache()

        # Add to total loss for the epoch
        total_loss += loss.item()

    return total_loss / len(data_loader)


def train_one_epoch_with_criterion(model, optimizer, device, criterion, data_loader):
    model.train()
    optimizer.zero_grad()
    
    # Loop over batches in data_loader
    total_loss = 0.0
    for x_batch, y_batch in data_loader:
        # Move the batch to the correct device (GPU/CPU)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # Forward pass
        output = model(x_batch)  # output shape: (batch_size, seq_len, vocab_size)

        # Flatten output and target tensors
        output = output.view(-1, output.size(-1))  # Flatten to (batch_size * seq_len, vocab_size)
        y_batch = y_batch.view(-1)  # Flatten to (batch_size * seq_len)

        # Calculate loss
        loss = criterion(output, y_batch)  # Use padding token if needed
        
        # Backpropagation
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Zero gradients for the next batch
        optimizer.zero_grad()
        
        # Clear GPU memory (optional, but useful for large batches)
        torch.cuda.empty_cache()

        # Add to total loss for the epoch
        total_loss += loss.item()

    return total_loss / len(data_loader)


def generate_sonnet(model, tokenizer, start_sequence, max_length=256, device='cuda'):
    model.eval()
    generated = tokenizer.encode(start_sequence)  # encode starting text
    generated = torch.tensor(generated, device=device).unsqueeze(0)  # shape (1, seq_len)

    for _ in range(max_length - generated.size(1)):
        outputs = model(generated)  # shape: (1, seq_len, vocab_size)
        next_token_logits = outputs[0, -1, :]  # last token logits
        
        # Greedy decoding: pick token with highest probability
        next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
        
        # Append to generated tokens
        generated = torch.cat([generated, next_token_id], dim=1)
        
        # Optional: stop if EOS token generated (if you have one)
        # if next_token_id.item() == tokenizer.eos_token_id:
        #     break
    
    # Decode tokens back to text
    generated_text = tokenizer.decode(generated[0].tolist())
    return generated_text


def top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to keep at least one token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value
    return logits

def generate_sonnet_sampling(model, sp, start_sequence, max_length=256, device='cuda', top_k=50):
    model.eval()
    generated = sp.encode_as_ids(start_sequence)
    generated = torch.tensor(generated, device=device).unsqueeze(0)

    for _ in range(max_length - generated.size(1)):
        outputs = model(generated)
        next_token_logits = outputs[0, -1, :]

        # Use this in your generation loop before sampling
        logits = top_p_filtering(next_token_logits, top_p=0.9)
        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).unsqueeze(0)

        generated = torch.cat([generated, next_token_id], dim=1)

        # Optional: break on EOS token if you have one

    return sp.decode_ids(generated[0].tolist())


def clean_generated_text(text, pad_token='<PAD>'):
    text = text.split(pad_token)[0].strip()
    text = text.replace('<LINE>', '\n').replace('<PAD>', '').strip()
    return text