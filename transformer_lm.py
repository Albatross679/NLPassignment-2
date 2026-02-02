# transformer_lm.py

import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
from utils import Indexer


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


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.emb = nn.Embedding(num_positions, d_model)

    def forward(self, x):
        """
        :param x: [batch_size, seq_len, d_model] or [seq_len, d_model]
        :return: tensor with positional embeddings added
        """
        if x.dim() == 3:
            seq_len = x.shape[1]
            indices = torch.arange(seq_len, device=x.device)
            pos_emb = self.emb(indices).unsqueeze(0)  # [1, seq_len, d_model]
            return self.dropout(x + pos_emb)
        else:
            seq_len = x.shape[0]
            indices = torch.arange(seq_len, device=x.device)
            pos_emb = self.emb(indices)  # [seq_len, d_model]
            return self.dropout(x + pos_emb)


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_layers, num_heads, dropout,
                 layer_norm_type="post", layer_norm_eps=1e-5):
        super().__init__()
        self.d_model = d_model
        self.num_positions = num_positions

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, num_positions, dropout)

        # Determine norm_first based on layer_norm_type
        # "pre" = norm_first=True (Pre-LN), "post" = norm_first=False (Post-LN)
        norm_first = (layer_norm_type == "pre")

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_internal,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
            layer_norm_eps=layer_norm_eps
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        :param x: [batch_size, seq_len] input token indices
        :return: [batch_size, seq_len, vocab_size] log probabilities
        """
        seq_len = x.shape[1]

        # Embed tokens
        x = self.embedding(x) * math.sqrt(self.d_model)  # [batch, seq_len, d_model]

        # Add positional encoding
        x = self.pos_encoding(x)

        # Generate causal mask (upper triangular)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)

        # Apply transformer with causal mask
        x = self.transformer(x, mask=causal_mask, is_causal=True)

        # Project to vocabulary
        logits = self.output_proj(x)  # [batch, seq_len, vocab_size]
        log_probs = torch.log_softmax(logits, dim=-1)

        return log_probs


class NeuralLanguageModel(LanguageModel):
    def __init__(self, model, vocab_index, device, num_positions):
        self.model = model
        self.vocab_index = vocab_index
        self.device = device
        self.num_positions = num_positions

    def get_next_char_log_probs(self, context):
        self.model.eval()
        with torch.no_grad():
            # Use space as start token, then add context
            # Truncate context if too long (keep last num_positions-1 chars)
            if len(context) >= self.num_positions:
                context = context[-(self.num_positions - 1):]

            # Convert context to indices (prepend space as start token)
            full_context = ' ' + context
            indices = [self.vocab_index.index_of(c) for c in full_context]
            input_tensor = torch.LongTensor(indices).unsqueeze(0).to(self.device)

            # Forward pass
            log_probs = self.model(input_tensor)

            # Return log probs for next character (after last position)
            return log_probs[0, -1, :].cpu().numpy()


def compute_gradient_norm(model):
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def evaluate_perplexity(model, text, vocab_index, device, context_size):
    """Compute perplexity on text."""
    model.eval()
    total_loss = 0.0
    total_chars = 0

    loss_fn = nn.NLLLoss(reduction='sum')

    with torch.no_grad():
        # Process text in chunks
        for i in range(0, len(text) - context_size, context_size):
            chunk = ' ' + text[i:i + context_size]  # prepend space as start
            if len(chunk) < 2:
                continue

            indices = [vocab_index.index_of(c) for c in chunk]
            input_tensor = torch.LongTensor(indices[:-1]).unsqueeze(0).to(device)
            target_tensor = torch.LongTensor(indices[1:]).to(device)

            log_probs = model(input_tensor)
            loss = loss_fn(log_probs[0], target_tensor)

            total_loss += loss.item()
            total_chars += len(target_tensor)

    model.train()
    avg_loss = total_loss / total_chars if total_chars > 0 else float('inf')
    return math.exp(avg_loss)


class TransformerLayerLM(nn.Module):
    """
    Custom Transformer layer for language modeling with multi-head attention.

    Supports three attention types:
    - standard: Scaled dot-product attention
    - relative_position: Attention with learned relative position biases
    - alibi: Attention with Linear Biases (no positional encoding needed)
    """

    def __init__(self, d_model, d_internal, num_heads=4, dropout=0.1,
                 attention_type="standard", max_relative_position=128,
                 num_positions=128, layer_norm_type="pre", layer_norm_eps=1e-5):
        """
        Args:
            d_model: Model dimension (embedding size)
            d_internal: Feedforward hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            attention_type: "standard", "relative_position", or "alibi"
            max_relative_position: Max relative position for relative position attention
            num_positions: Max sequence length (for ALiBi bias pre-computation)
            layer_norm_type: "pre" or "post" layer normalization
            layer_norm_eps: Epsilon for layer normalization
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.attention_type = attention_type
        self.max_relative_position = max_relative_position
        self.num_positions = num_positions
        self.layer_norm_type = layer_norm_type

        # Multi-head attention projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Feed-forward network
        self.ff1 = nn.Linear(d_model, d_internal)
        self.ff2 = nn.Linear(d_internal, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Relative position embeddings (for relative_position attention)
        if attention_type == "relative_position":
            num_relative_positions = 2 * max_relative_position - 1
            self.relative_position_embeddings = nn.Embedding(num_relative_positions, self.head_dim)

        # ALiBi slopes (pre-computed, not learned)
        if attention_type == "alibi":
            # Compute slopes for each head: 2^(-8*(i+1)/n) for head i of n heads
            slopes = torch.tensor([
                2.0 ** (-8.0 * (i + 1) / num_heads) for i in range(num_heads)
            ])
            self.register_buffer('alibi_slopes', slopes)  # [num_heads]

    def _compute_alibi_bias(self, seq_len, device):
        """Compute ALiBi position bias matrix."""
        # positions[i, j] = j - i (relative distance)
        positions = torch.arange(seq_len, device=device)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq_len, seq_len]

        # Multiply by slopes for each head: [num_heads, seq_len, seq_len]
        # distance is broadcast: [1, seq_len, seq_len] * [num_heads, 1, 1]
        alibi_bias = -torch.abs(distance.float()).unsqueeze(0) * self.alibi_slopes.view(-1, 1, 1)
        return alibi_bias

    def _compute_relative_position_bias(self, Q, seq_len, device):
        """Compute relative position bias for attention scores."""
        # Compute relative positions matrix
        positions = torch.arange(seq_len, device=device)
        rel_positions = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq_len, seq_len]

        # Clip and shift to valid embedding indices
        rel_positions = rel_positions.clamp(-self.max_relative_position + 1, self.max_relative_position - 1)
        rel_positions = rel_positions + self.max_relative_position - 1  # [0, 2*max-2]

        # Get relative position embeddings: [seq_len, seq_len, head_dim]
        rel_emb = self.relative_position_embeddings(rel_positions)

        # Q: [batch_size, num_heads, seq_len, head_dim]
        # Compute Q @ rel_emb for each position
        # Result: [batch_size, num_heads, seq_len, seq_len]
        rel_bias = torch.einsum('bhqd,qkd->bhqk', Q, rel_emb) / math.sqrt(self.head_dim)
        return rel_bias

    def forward(self, x, mask=None):
        """
        Forward pass through the transformer layer.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional causal mask [seq_len, seq_len]

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Pre-norm
        if self.layer_norm_type == "pre":
            normed = self.norm1(x)
        else:
            normed = x

        # Multi-head attention projections
        Q = self.W_q(normed)  # [batch, seq, d_model]
        K = self.W_k(normed)
        V = self.W_v(normed)

        # Reshape for multi-head: [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # [batch, heads, seq, head_dim] @ [batch, heads, head_dim, seq] -> [batch, heads, seq, seq]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply attention-type-specific biases
        if self.attention_type == "relative_position":
            rel_bias = self._compute_relative_position_bias(Q, seq_len, x.device)
            scores = scores + rel_bias
        elif self.attention_type == "alibi":
            alibi_bias = self._compute_alibi_bias(seq_len, x.device)  # [num_heads, seq, seq]
            scores = scores + alibi_bias.unsqueeze(0)  # [1, num_heads, seq, seq]

        # Apply causal mask
        if mask is not None:
            # mask: [seq, seq] -> [1, 1, seq, seq] for broadcasting
            scores = scores + mask.unsqueeze(0).unsqueeze(0)

        # Softmax and attention
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [batch, heads, seq, head_dim]

        # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.W_o(attn_output)
        attn_output = self.dropout(attn_output)

        # First residual connection
        x = x + attn_output

        # Post-norm after attention
        if self.layer_norm_type == "post":
            x = self.norm1(x)

        # Pre-norm before FFN
        if self.layer_norm_type == "pre":
            normed = self.norm2(x)
        else:
            normed = x

        # Feed-forward network
        ff_output = self.ff2(self.dropout(torch.relu(self.ff1(normed))))
        ff_output = self.dropout(ff_output)

        # Second residual connection
        x = x + ff_output

        # Post-norm after FFN
        if self.layer_norm_type == "post":
            x = self.norm2(x)

        return x


class TransformerLMCustom(nn.Module):
    """
    Transformer Language Model with custom attention mechanisms.

    Supports standard, relative position, and ALiBi attention types.
    Uses TransformerLayerLM instead of nn.TransformerEncoder.
    """

    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_layers, num_heads, dropout,
                 attention_type="standard", use_positional_encoding=True,
                 max_relative_position=128, layer_norm_type="pre", layer_norm_eps=1e-5):
        """
        Args:
            vocab_size: Vocabulary size
            num_positions: Maximum sequence length
            d_model: Model dimension
            d_internal: Feedforward hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            attention_type: "standard", "relative_position", or "alibi"
            use_positional_encoding: Whether to add positional encoding (disable for ALiBi)
            max_relative_position: Max relative position for relative position attention
            layer_norm_type: "pre" or "post" layer normalization
            layer_norm_eps: Epsilon for layer normalization
        """
        super().__init__()
        self.d_model = d_model
        self.num_positions = num_positions
        self.use_positional_encoding = use_positional_encoding

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding (optional - disabled for ALiBi)
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model, num_positions, dropout)

        # Stack of custom transformer layers
        self.layers = nn.ModuleList([
            TransformerLayerLM(
                d_model=d_model,
                d_internal=d_internal,
                num_heads=num_heads,
                dropout=dropout,
                attention_type=attention_type,
                max_relative_position=max_relative_position,
                num_positions=num_positions,
                layer_norm_type=layer_norm_type,
                layer_norm_eps=layer_norm_eps
            )
            for _ in range(num_layers)
        ])

        # Final layer norm (for pre-LN architecture)
        if layer_norm_type == "pre":
            self.final_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

        self.layer_norm_type = layer_norm_type

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input token indices [batch_size, seq_len]

        Returns:
            Log probabilities [batch_size, seq_len, vocab_size]
        """
        seq_len = x.shape[1]

        # Embed tokens
        x = self.embedding(x) * math.sqrt(self.d_model)  # [batch, seq_len, d_model]

        # Add positional encoding if enabled
        if self.use_positional_encoding:
            x = self.pos_encoding(x)

        # Generate causal mask (upper triangular with -inf)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=x.device),
            diagonal=1
        )

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask=causal_mask)

        # Final layer norm for pre-LN
        if self.layer_norm_type == "pre":
            x = self.final_norm(x)

        # Project to vocabulary
        logits = self.output_proj(x)  # [batch, seq_len, vocab_size]
        log_probs = torch.log_softmax(logits, dim=-1)

        return log_probs


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    # ===== HARDCODED HYPERPARAMETERS (for autograder reliability) =====
    # These values are from the proven part2_standard configuration

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
    num_positions = 128
    d_model = 128
    d_internal = 128
    num_layers = 2
    num_heads = 4
    dropout = 0.1
    context_size = 64

    # Training hyperparameters (hardcoded)
    num_epochs = 20
    batch_size = 64
    learning_rate = 0.001

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
    weight_init_method = "xavier_uniform"

    # Build hp_cfg dict for helper functions
    hp_cfg = {
        'learning_rate': learning_rate,
        'optimizer_type': 'Adam',
        'weight_init_enabled': weight_init_enabled,
        'weight_init_method': weight_init_method,
    }

    # Create a minimal run directory
    run_dir = Path(__file__).parent / 'output' / 'part2_run'
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment output directory: {run_dir}")

    # Create model (using relative position attention with custom transformer)
    model = TransformerLMCustom(vocab_size, num_positions, d_model, d_internal, num_layers, num_heads, dropout,
                                attention_type="relative_position",
                                layer_norm_type=layer_norm_type, layer_norm_eps=layer_norm_eps)

    # Apply weight initialization
    init_weights(model, hp_cfg)

    model = model.to(device)
    model.train()

    # Create optimizer (simple Adam with hardcoded learning rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = None  # No scheduler for simplicity

    loss_fn = nn.NLLLoss()

    # Prepare training data - create sequences
    train_indices = [vocab_index.index_of(c) for c in train_text]

    start_time = time.time()

    # Create batches of sequences
    num_sequences = (len(train_indices) - 1) // context_size

    # Early stopping state
    best_dev_perplexity = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_chars = 0
        grad_norm_sum = 0.0
        num_batches = 0

        # Shuffle starting positions for diversity (use seed for reproducibility)
        positions = list(range(0, len(train_indices) - context_size - 1, context_size // 2))
        np.random.seed(seed + epoch)
        np.random.shuffle(positions)

        # Process in batches
        for batch_start in range(0, len(positions), batch_size):
            batch_positions = positions[batch_start:batch_start + batch_size]
            if len(batch_positions) == 0:
                continue

            # Build batch
            batch_inputs = []
            batch_targets = []

            for pos in batch_positions:
                # Input: positions 0 to context_size-1
                # Target: positions 1 to context_size
                seq = train_indices[pos:pos + context_size + 1]
                if len(seq) < context_size + 1:
                    continue
                batch_inputs.append(seq[:-1])
                batch_targets.append(seq[1:])

            if len(batch_inputs) == 0:
                continue

            input_tensor = torch.LongTensor(batch_inputs).to(device)
            target_tensor = torch.LongTensor(batch_targets).to(device)

            optimizer.zero_grad()

            # Forward pass
            log_probs = model(input_tensor)

            # Compute loss
            # log_probs: [batch, seq_len, vocab_size]
            # target: [batch, seq_len]
            loss = loss_fn(log_probs.view(-1, vocab_size), target_tensor.view(-1))

            # Backward pass
            loss.backward()

            # Apply gradient clipping if enabled
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)

            # Track gradient norm
            batch_grad_norm = compute_gradient_norm(model)
            grad_norm_sum += batch_grad_norm
            num_batches += 1

            optimizer.step()

            total_loss += loss.item() * target_tensor.numel()
            total_chars += target_tensor.numel()

        # Step scheduler if enabled
        if scheduler is not None:
            scheduler.step()

        # Compute epoch metrics
        avg_loss = total_loss / total_chars if total_chars > 0 else float('inf')
        train_perplexity = math.exp(avg_loss)
        avg_grad_norm = grad_norm_sum / num_batches if num_batches > 0 else 0.0
        dev_perplexity = evaluate_perplexity(model, dev_text, vocab_index, device, context_size)
        elapsed_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Train PPL: {train_perplexity:.2f}, Dev PPL: {dev_perplexity:.2f}, LR: {current_lr:.6f}")

        # Early stopping check (for perplexity, lower is better)
        if use_early_stopping:
            if dev_perplexity < best_dev_perplexity - min_delta:
                best_dev_perplexity = dev_perplexity
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

    model.eval()
    return NeuralLanguageModel(model, vocab_index, device, num_positions)
