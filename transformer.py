# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
import matplotlib.pyplot as plt
from typing import List
from utils import *


def create_optimizer(model, hp_cfg):
    """Factory function to create optimizer based on hyperparameters config."""
    opt_type = hp_cfg.get('optimizer_type', 'Adam')
    lr = hp_cfg.get('learning_rate', 0.001)
    weight_decay = hp_cfg.get('optimizer_weight_decay', 0.0)

    if opt_type == 'Adam':
        betas = tuple(hp_cfg.get('optimizer_betas', [0.9, 0.999]))
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    elif opt_type == 'SGD':
        momentum = hp_cfg.get('optimizer_momentum', 0.9)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_type == 'AdamW':
        betas = tuple(hp_cfg.get('optimizer_betas', [0.9, 0.999]))
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    else:
        return optim.Adam(model.parameters(), lr=lr)


def create_scheduler(optimizer, hp_cfg, num_epochs):
    """Factory function to create learning rate scheduler based on hyperparameters config."""
    if not hp_cfg.get('scheduler_enabled', False):
        return None
    sched_type = hp_cfg.get('scheduler_type', 'cosine')
    if sched_type == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=hp_cfg.get('scheduler_min_lr', 1e-6)
        )
    elif sched_type == 'step':
        return StepLR(
            optimizer,
            step_size=hp_cfg.get('scheduler_step_size', 10),
            gamma=hp_cfg.get('scheduler_gamma', 0.1)
        )
    elif sched_type == 'exponential':
        return ExponentialLR(
            optimizer,
            gamma=hp_cfg.get('scheduler_gamma', 0.95)
        )
    return None


def init_weights(model, hp_cfg):
    """Apply weight initialization based on hyperparameters config."""
    if not hp_cfg.get('weight_init_enabled', False):
        return
    method = hp_cfg.get('weight_init_method', 'uniform')
    min_val = hp_cfg.get('weight_init_min', -0.1)
    max_val = hp_cfg.get('weight_init_max', 0.1)

    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue  # Skip biases and 1D params
        if method == 'uniform':
            nn.init.uniform_(param, min_val, max_val)
        elif method == 'xavier_uniform':
            nn.init.xavier_uniform_(param)
        elif method == 'xavier_normal':
            nn.init.xavier_normal_(param)
        elif method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(param, nonlinearity='relu')
        elif method == 'kaiming_normal':
            nn.init.kaiming_normal_(param, nonlinearity='relu')


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers,
                 attention_type="standard", use_positional_encoding=True,
                 max_relative_position=20, alibi_slope=0.125,
                 layer_norm_type="post", layer_norm_eps=1e-5):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        :param attention_type: Type of attention ("standard", "relative_position", "alibi")
        :param use_positional_encoding: Whether to add positional encoding to embeddings
        :param max_relative_position: Maximum relative position for relative position attention
        :param alibi_slope: Slope for ALiBi attention bias
        :param layer_norm_type: Type of layer normalization ("pre", "post", or "none")
        :param layer_norm_eps: Epsilon for layer normalization
        """
        super().__init__()
        # Character embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding (optional)
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model, num_positions)

        # Stack of TransformerLayers using nn.ModuleList
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, d_internal, attention_type=attention_type,
                           max_relative_position=max_relative_position,
                           alibi_slope=alibi_slope, num_positions=num_positions,
                           layer_norm_type=layer_norm_type, layer_norm_eps=layer_norm_eps)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, num_classes)

        # Store num_positions for causal mask
        self.num_positions = num_positions

    def forward(self, indices, use_causal_mask=True):
        """

        :param indices: Input indices - either [seq_len] for unbatched or [batch_size, seq_len] for batched
        :param use_causal_mask: Whether to apply causal mask (default True for BEFORE task)
        :return: A tuple of the softmax log probabilities and a list of the attention
        maps you use in your layers. For unbatched: [seq_len, num_classes] and [seq_len, seq_len] maps.
        For batched: [batch_size, seq_len, num_classes] and [batch_size, seq_len, seq_len] maps.
        """
        # Detect batched vs unbatched input
        is_batched = indices.dim() == 2
        if not is_batched:
            indices = indices.unsqueeze(0)  # [1, seq_len]

        batch_size, seq_len = indices.shape

        # Embed input indices
        x = self.embedding(indices)  # [batch_size, seq_len, d_model]

        # Add positional encoding if enabled
        if self.use_positional_encoding:
            x = self.positional_encoding(x)  # [batch_size, seq_len, d_model]

        # Generate causal mask if needed
        mask = None
        if use_causal_mask:
            # Upper triangular matrix with -inf above diagonal (tokens can't attend to future)
            # Shape [seq_len, seq_len] - broadcasts to [batch_size, seq_len, seq_len]
            mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)

        # Pass through transformer layers
        attn_maps = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask=mask)
            attn_maps.append(attn_weights)

        # Output projection and log softmax
        logits = self.output_proj(x)  # [batch_size, seq_len, num_classes]
        log_probs = torch.log_softmax(logits, dim=-1)  # [batch_size, seq_len, num_classes]

        # Squeeze if unbatched input for backward compatibility
        if not is_batched:
            log_probs = log_probs.squeeze(0)  # [seq_len, num_classes]
            attn_maps = [attn.squeeze(0) for attn in attn_maps]  # [seq_len, seq_len] each

        return log_probs, attn_maps


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal, attention_type="standard",
                 max_relative_position=20, alibi_slope=0.125, num_positions=20,
                 layer_norm_type="post", layer_norm_eps=1e-5):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        :param attention_type: Type of attention ("standard", "relative_position", "alibi")
        :param max_relative_position: Maximum relative position for relative position attention
        :param alibi_slope: Slope for ALiBi attention bias
        :param num_positions: Maximum sequence length (for pre-computing ALiBi bias)
        :param layer_norm_type: Type of layer normalization ("pre", "post", or "none")
        :param layer_norm_eps: Epsilon for layer normalization
        """
        super().__init__()
        # Attention projections
        self.W_q = nn.Linear(d_model, d_internal)
        self.W_k = nn.Linear(d_model, d_internal)
        self.W_v = nn.Linear(d_model, d_internal)
        self.W_o = nn.Linear(d_internal, d_model)

        # Feed-forward network
        self.ff1 = nn.Linear(d_model, d_internal)
        self.ff2 = nn.Linear(d_internal, d_model)

        # Layer normalization
        self.layer_norm_type = layer_norm_type
        if layer_norm_type in ("pre", "post"):
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Store d_internal for scaling
        self.d_internal = d_internal
        self.attention_type = attention_type
        self.max_relative_position = max_relative_position
        self.num_positions = num_positions

        # Relative position embeddings (for relative_position attention)
        if attention_type == "relative_position":
            # Embeddings for relative positions from -(max-1) to +(max-1)
            num_relative_positions = 2 * max_relative_position - 1
            self.relative_position_embeddings = nn.Embedding(num_relative_positions, d_internal)

        # ALiBi bias (pre-computed, not learned)
        if attention_type == "alibi":
            # Pre-compute ALiBi bias matrix: -slope * |i - j| for each (i, j) position
            positions = torch.arange(num_positions)
            # distance[i, j] = j - i (positive means j is to the right of i)
            distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # [num_positions, num_positions]
            alibi_bias = -alibi_slope * torch.abs(distance.float())  # [num_positions, num_positions]
            # Register as buffer (not a parameter, but moves with model to device)
            self.register_buffer('alibi_bias', alibi_bias)

    def forward(self, input_vecs, mask=None):
        """
        :param input_vecs: tensor of shape [seq_len, d_model] or [batch_size, seq_len, d_model]
        :param mask: optional attention mask of shape [seq_len, seq_len]
        :return: tuple of (output_vecs, attention_weights)
        """
        # Handle both batched and unbatched inputs
        is_batched = input_vecs.dim() == 3
        if is_batched:
            batch_size, seq_len, _ = input_vecs.shape
        else:
            seq_len = input_vecs.size(0)

        # Pre-norm: apply layer norm before attention
        if self.layer_norm_type == "pre":
            normed_input = self.norm1(input_vecs)
        else:
            normed_input = input_vecs

        # Self-attention
        # For batched: [batch_size, seq_len, d_internal], for unbatched: [seq_len, d_internal]
        Q = self.W_q(normed_input)
        K = self.W_k(normed_input)
        V = self.W_v(normed_input)

        # Scaled dot-product attention
        # transpose(-2, -1) works for both batched and unbatched
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_internal)

        # Apply attention-type-specific biases
        if self.attention_type == "relative_position":
            # Compute relative positions
            positions = torch.arange(seq_len, device=input_vecs.device)
            # rel_positions[i, j] = j - i (relative position of key j from query i)
            rel_positions = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq_len, seq_len]
            # Clip to valid range and shift to positive indices
            rel_positions = rel_positions.clamp(-self.max_relative_position + 1, self.max_relative_position - 1)
            rel_positions = rel_positions + self.max_relative_position - 1  # Shift to [0, 2*max-2]

            # Get relative position embeddings for each (query, key) pair
            rel_emb = self.relative_position_embeddings(rel_positions)  # [seq_len, seq_len, d_internal]

            # Compute relative position bias: Q @ rel_emb for each query position
            if is_batched:
                # Q: [batch_size, seq_len, d_internal], rel_emb: [seq_len, seq_len, d_internal]
                rel_bias = torch.einsum('bqd,qkd->bqk', Q, rel_emb) / np.sqrt(self.d_internal)
            else:
                # Q: [seq_len, d_internal], rel_emb: [seq_len, seq_len, d_internal]
                rel_bias = torch.einsum('qd,qkd->qk', Q, rel_emb) / np.sqrt(self.d_internal)
            scores = scores + rel_bias

        elif self.attention_type == "alibi":
            # Add pre-computed ALiBi bias (truncate if seq_len < num_positions)
            # ALiBi bias [seq_len, seq_len] broadcasts to batched scores
            scores = scores + self.alibi_bias[:seq_len, :seq_len]

        # Apply mask if provided (for causal attention)
        # mask [seq_len, seq_len] broadcasts to [batch_size, seq_len, seq_len] for batched
        if mask is not None:
            scores = scores + mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = self.W_o(attn_output)

        # First residual connection
        x = input_vecs + attn_output

        # Post-norm after attention: apply layer norm after residual
        if self.layer_norm_type == "post":
            x = self.norm1(x)

        # Pre-norm: apply layer norm before FFN
        if self.layer_norm_type == "pre":
            normed_x = self.norm2(x)
        else:
            normed_x = x

        # Feed-forward network
        ff_output = self.ff2(torch.relu(self.ff1(normed_x)))

        # Second residual connection
        output = x + ff_output

        # Post-norm after FFN: apply layer norm after residual
        if self.layer_norm_type == "post":
            output = self.norm2(output)

        return output, attn_weights


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise (deprecated - now auto-detected)
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Auto-detect batching based on input dimensions
        is_batched = x.dim() == 3

        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.arange(input_size, dtype=torch.long, device=x.device)
        if is_batched or self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


def compute_gradient_norm(model):
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def collate_examples(examples, device):
    """Stack examples into batched tensors."""
    input_batch = torch.stack([ex.input_tensor for ex in examples]).to(device)
    output_batch = torch.stack([ex.output_tensor for ex in examples]).to(device)
    return input_batch, output_batch


def evaluate_accuracy(model, examples, device, batch_size=32, use_causal_mask=False):
    """Compute accuracy on a set of examples with optional batching."""
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            inputs = torch.stack([ex.input_tensor for ex in batch]).to(device)
            targets = torch.stack([ex.output_tensor for ex in batch])

            log_probs, _ = model.forward(inputs, use_causal_mask=use_causal_mask)
            preds = torch.argmax(log_probs, dim=-1).cpu()

            correct += (preds == targets).sum().item()
            total += targets.numel()
    model.train()
    return correct / total


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    # ===== HARDCODED HYPERPARAMETERS (for autograder reliability) =====
    # These values are from the proven part1_standard_3layer configuration

    # Random seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model architecture (hardcoded)
    vocab_size = 27
    num_positions = 20
    d_model = 64
    d_internal = 64
    num_classes = 3
    num_layers = 3

    # Training hyperparameters (hardcoded)
    num_epochs = 30
    learning_rate = 0.001
    batch_size = 32

    # Gradient clipping (disabled)
    use_grad_clip = False
    grad_clip_max_norm = 1.0

    # Early stopping
    use_early_stopping = True
    patience = 5
    min_delta = 0.001

    # Layer normalization
    layer_norm_type = "pre"
    layer_norm_eps = 1e-5

    # Weight initialization
    weight_init_enabled = True
    weight_init_method = "uniform"
    weight_init_min = -0.1
    weight_init_max = 0.1

    # Build hp_cfg dict for helper functions
    hp_cfg = {
        'learning_rate': learning_rate,
        'optimizer_type': 'Adam',
        'weight_init_enabled': weight_init_enabled,
        'weight_init_method': weight_init_method,
        'weight_init_min': weight_init_min,
        'weight_init_max': weight_init_max,
    }

    # Create a minimal run directory
    run_dir = Path(__file__).parent / 'output' / 'part1_run'
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment output directory: {run_dir}")

    # Create model (using relative position attention)
    model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers,
                        attention_type="relative_position",
                        layer_norm_type=layer_norm_type, layer_norm_eps=layer_norm_eps)

    # Apply weight initialization
    init_weights(model, hp_cfg)

    model = model.to(device)
    model.zero_grad()
    model.train()

    # Create optimizer (simple Adam with hardcoded learning rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = None  # No scheduler for simplicity

    loss_fcn = nn.NLLLoss()

    start_time = time.time()

    # Early stopping state
    best_dev_accuracy = 0.0
    epochs_without_improvement = 0

    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        correct_this_epoch = 0
        total_this_epoch = 0
        grad_norm_sum = 0.0
        num_batches_processed = 0
        random.seed(seed + t)  # Use seed for reproducibility
        ex_idxs = list(range(len(train)))
        random.shuffle(ex_idxs)

        # Process in batches
        num_batches = (len(ex_idxs) + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(ex_idxs))
            batch_examples = [train[ex_idxs[i]] for i in range(start, end)]

            input_batch, output_batch = collate_examples(batch_examples, device)

            model.zero_grad()

            # Forward pass with causal mask (BEFORE task)
            log_probs, _ = model.forward(input_batch, use_causal_mask=True)

            # Track training accuracy
            predictions = torch.argmax(log_probs, dim=-1)
            correct_this_epoch += (predictions == output_batch).sum().item()
            total_this_epoch += output_batch.numel()

            # Compute loss over all positions
            # Reshape for NLLLoss: [batch*seq, classes] vs [batch*seq]
            loss = loss_fcn(log_probs.view(-1, num_classes), output_batch.view(-1))

            # Backward pass
            loss.backward()

            # Apply gradient clipping if enabled
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)

            # Track gradient norm (after backward, before optimizer step)
            batch_grad_norm = compute_gradient_norm(model)
            grad_norm_sum += batch_grad_norm
            num_batches_processed += 1

            optimizer.step()

            loss_this_epoch += loss.item()

        # Step scheduler if enabled
        if scheduler is not None:
            scheduler.step()

        # Compute epoch metrics
        avg_loss = loss_this_epoch / num_batches_processed
        train_accuracy = correct_this_epoch / total_this_epoch
        avg_grad_norm = grad_norm_sum / num_batches_processed
        dev_accuracy = evaluate_accuracy(model, dev, device, batch_size=batch_size, use_causal_mask=True)
        elapsed_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {t+1}, Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}, Dev Acc: {dev_accuracy:.4f}, LR: {current_lr:.6f}")

        # Early stopping check
        if use_early_stopping:
            if dev_accuracy > best_dev_accuracy + min_delta:
                best_dev_accuracy = dev_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {t+1} epochs")
                    break

    # Move model back to CPU for decode function compatibility
    model = model.to('cpu')
    model.eval()
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False, do_attention_normalization_test=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
        do_attention_normalization_test = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        if do_attention_normalization_test:
            normalizes = attention_normalization_test(attn_maps)
            print("%s normalization test on attention maps" % ("Passed" if normalizes else "Failed"))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))


def attention_normalization_test(attn_maps):
    """
    Tests that the attention maps sum to one over rows
    :param attn_maps: the list of attention maps
    :return:
    """
    for attn_map in attn_maps:
        total_prob_over_rows = torch.sum(attn_map, dim=1)
        if torch.any(total_prob_over_rows < 0.99).item() or torch.any(total_prob_over_rows > 1.01).item():
            print("Failed normalization test: probabilities not sum to 1.0 over rows")
            print("Total probability over rows:", total_prob_over_rows)
            return False
    return True
