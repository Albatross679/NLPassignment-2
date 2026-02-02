# metrics_logger.py
"""
Unified TensorBoard metrics logging utility.

Provides a MetricsLogger class that wraps TensorBoard SummaryWriter with
configurable metrics logging for hyperparameters, per-batch metrics,
per-epoch metrics, and final inference outputs.
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class MetricsLogger:
    """
    Unified metrics logger wrapping TensorBoard SummaryWriter.

    Supports configurable logging of:
    - Hyperparameters (logged once at start)
    - Per-batch metrics (loss, gradient_norm)
    - Per-epoch metrics (task-specific: accuracy for classification, perplexity for LM)
    - Final inference outputs (attention maps, log_probs heatmaps, predictions as text)
    - Optional histograms for weights and gradients
    """

    def __init__(
        self,
        config: Dict[str, Any],
        experiment_name: str,
        task_type: str = 'classification',
        base_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize the MetricsLogger.

        Args:
            config: TensorBoard configuration dict from tensorboard.json
            experiment_name: Name for this experiment (e.g., 'part1', 'part2')
            task_type: Either 'classification' or 'language_modeling'
            base_dir: Base directory for log files (defaults to current directory)
            output_dir: Experiment output directory (if provided, tensorboard logs go here)
        """
        self.config = config
        self.experiment_name = experiment_name
        self.task_type = task_type
        self.base_dir = base_dir or Path.cwd()
        self.output_dir = Path(output_dir) if output_dir else None

        self.enabled = config.get('enabled', False) and TENSORBOARD_AVAILABLE
        self.writer = None
        self.global_step = 0
        self.batch_step = 0

        if self.enabled:
            self._init_writer()

    def _init_writer(self):
        """Initialize TensorBoard SummaryWriter."""
        flush_secs = self.config.get('flush_secs', 120)
        configured_log_dir = self.config.get('log_dir', 'runs')

        # Use output_dir/tensorboard if provided, otherwise use configured log_dir
        if self.output_dir:
            log_dir = self.output_dir / 'tensorboard'
        else:
            log_dir = self.base_dir / configured_log_dir
            run_name = f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            log_dir = log_dir / run_name

        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(
                log_dir=log_dir,
                flush_secs=flush_secs
            )
            print(f"TensorBoard logging to: {log_dir}")
        except Exception as e:
            print(f"Failed to initialize TensorBoard: {e}")
            self.enabled = False

    def log_batch(
        self,
        loss: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        **kwargs
    ):
        """
        Log per-batch metrics.

        Args:
            loss: Batch loss value
            gradient_norm: Gradient norm after backward pass
            **kwargs: Additional batch metrics
        """
        if not self.enabled or self.writer is None:
            return

        if not self.config.get('per_batch_enabled', True):
            return

        frequency = self.config.get('per_batch_frequency', 10)

        self.batch_step += 1

        # Only log at specified frequency
        if self.batch_step % frequency != 0:
            return

        if loss is not None and self.config.get('per_batch_loss', True):
            self.writer.add_scalar('Batch/loss', loss, self.batch_step)

        if gradient_norm is not None and self.config.get('per_batch_gradient_norm', True):
            self.writer.add_scalar('Batch/gradient_norm', gradient_norm, self.batch_step)

        # Log any additional metrics
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'Batch/{key}', value, self.batch_step)

    def log_epoch(
        self,
        epoch: int,
        train_loss: Optional[float] = None,
        train_accuracy: Optional[float] = None,
        train_perplexity: Optional[float] = None,
        dev_accuracy: Optional[float] = None,
        dev_perplexity: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        learning_rate: Optional[float] = None,
        elapsed_time: Optional[float] = None,
        **kwargs
    ):
        """
        Log per-epoch metrics.

        Automatically routes metrics based on task_type:
        - 'classification': logs accuracy, ignores perplexity
        - 'language_modeling': logs perplexity, ignores accuracy

        Args:
            epoch: Current epoch number (1-indexed)
            train_loss: Average training loss
            train_accuracy: Training accuracy (classification only)
            train_perplexity: Training perplexity (language modeling only)
            dev_accuracy: Dev set accuracy (classification only)
            dev_perplexity: Dev set perplexity (language modeling only)
            gradient_norm: Average gradient norm
            learning_rate: Current learning rate
            elapsed_time: Time elapsed since training start
            **kwargs: Additional epoch metrics
        """
        if not self.enabled or self.writer is None:
            return

        if not self.config.get('per_epoch_enabled', True):
            return

        # Train loss (common to both tasks)
        if train_loss is not None and self.config.get('per_epoch_train_loss', True):
            self.writer.add_scalar('Loss/train', train_loss, epoch)

        # Task-specific metrics
        if self.task_type == 'classification':
            if train_accuracy is not None and self.config.get('per_epoch_train_accuracy', True):
                self.writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            if dev_accuracy is not None and self.config.get('per_epoch_dev_accuracy', True):
                self.writer.add_scalar('Accuracy/dev', dev_accuracy, epoch)
        elif self.task_type == 'language_modeling':
            if train_perplexity is not None and self.config.get('per_epoch_train_perplexity', True):
                self.writer.add_scalar('Perplexity/train', train_perplexity, epoch)
            if dev_perplexity is not None and self.config.get('per_epoch_dev_perplexity', True):
                self.writer.add_scalar('Perplexity/dev', dev_perplexity, epoch)

        # Common metrics
        if gradient_norm is not None and self.config.get('per_epoch_gradient_norm', True):
            self.writer.add_scalar('Gradient/norm', gradient_norm, epoch)

        if learning_rate is not None and self.config.get('per_epoch_learning_rate', True):
            self.writer.add_scalar('LearningRate', learning_rate, epoch)

        if elapsed_time is not None and self.config.get('per_epoch_elapsed_time', True):
            self.writer.add_scalar('Time/elapsed_seconds', elapsed_time, epoch)

        # Log any additional metrics
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'Epoch/{key}', value, epoch)

    def log_inference(
        self,
        attention_maps: Optional[List[Any]] = None,
        log_probs: Optional[Any] = None,
        predictions: Optional[str] = None,
        input_text: Optional[str] = None,
        gold_labels: Optional[str] = None,
        sample_index: int = 0,
        generated_text: Optional[str] = None
    ):
        """
        Log final inference outputs.

        Args:
            attention_maps: List of attention weight tensors/arrays
            log_probs: Log probability tensor/array
            predictions: Predicted labels as string
            input_text: Input text for this sample
            gold_labels: Gold labels as string
            sample_index: Index of this sample
            generated_text: Generated text (for language modeling)
        """
        if not self.enabled or self.writer is None:
            return

        if not self.config.get('final_inference_enabled', True):
            return

        # Log attention maps as images
        if attention_maps is not None and self.config.get('attention_maps_enabled', True):
            num_samples = self.config.get('attention_maps_num_samples', 4)
            if sample_index < num_samples:
                colormap = self.config.get('attention_maps_colormap', 'viridis')
                self._log_attention_maps(attention_maps, sample_index, colormap)

        # Log log_probs as heatmap
        if log_probs is not None and self.config.get('log_probs_enabled', True):
            if self.config.get('log_probs_as_heatmap', True):
                self._log_log_probs_heatmap(log_probs, sample_index)

        # Log predictions as text
        if self.config.get('predictions_enabled', True):
            num_pred_samples = self.config.get('predictions_num_samples', 10)
            if sample_index < num_pred_samples:
                self._log_predictions_text(
                    predictions, input_text, gold_labels,
                    sample_index, generated_text
                )

    def _log_attention_maps(self, attention_maps: List[Any], sample_index: int, colormap: str):
        """Log attention maps as images."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import io
            from PIL import Image
            import torch

            for layer_idx, attn_map in enumerate(attention_maps):
                # Convert to numpy if tensor
                if hasattr(attn_map, 'detach'):
                    attn_map = attn_map.detach().cpu().numpy()
                elif hasattr(attn_map, 'cpu'):
                    attn_map = attn_map.cpu().numpy()

                # Normalize to [0, 1]
                attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

                # Create figure
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(attn_map, cmap=colormap, aspect='auto')
                ax.set_title(f'Sample {sample_index}, Layer {layer_idx}')
                ax.set_xlabel('Key position')
                ax.set_ylabel('Query position')
                plt.colorbar(im, ax=ax)

                # Convert to image tensor
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                img = Image.open(buf)
                img_array = np.array(img)
                plt.close(fig)

                # Convert to CHW format for TensorBoard
                if img_array.ndim == 3:
                    img_tensor = np.transpose(img_array[:, :, :3], (2, 0, 1))
                else:
                    img_tensor = img_array

                tag = f'Attention/sample_{sample_index}_layer_{layer_idx}'
                self.writer.add_image(tag, img_tensor, 0)

        except Exception as e:
            print(f"Warning: Failed to log attention maps: {e}")

    def _log_log_probs_heatmap(self, log_probs: Any, sample_index: int):
        """Log log probabilities as a heatmap image."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import io
            from PIL import Image

            # Convert to numpy if tensor
            if hasattr(log_probs, 'detach'):
                log_probs = log_probs.detach().cpu().numpy()
            elif hasattr(log_probs, 'cpu'):
                log_probs = log_probs.cpu().numpy()

            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(log_probs, cmap='RdYlBu', aspect='auto')
            ax.set_title(f'Log Probabilities - Sample {sample_index}')
            ax.set_xlabel('Class')
            ax.set_ylabel('Position')
            plt.colorbar(im, ax=ax)

            # Convert to image tensor
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            img_array = np.array(img)
            plt.close(fig)

            # Convert to CHW format
            if img_array.ndim == 3:
                img_tensor = np.transpose(img_array[:, :, :3], (2, 0, 1))
            else:
                img_tensor = img_array

            tag = f'LogProbs/sample_{sample_index}'
            self.writer.add_image(tag, img_tensor, 0)

        except Exception as e:
            print(f"Warning: Failed to log log_probs heatmap: {e}")

    def _log_predictions_text(
        self,
        predictions: Optional[str],
        input_text: Optional[str],
        gold_labels: Optional[str],
        sample_index: int,
        generated_text: Optional[str] = None
    ):
        """Log predictions as text."""
        try:
            text_parts = [f"**Sample {sample_index}**\n"]

            if input_text is not None:
                text_parts.append(f"Input: `{input_text}`\n")

            if gold_labels is not None:
                text_parts.append(f"Gold: `{gold_labels}`\n")

            if predictions is not None:
                text_parts.append(f"Pred: `{predictions}`\n")

            if generated_text is not None:
                text_parts.append(f"Generated: `{generated_text}`\n")

            text = "\n".join(text_parts)
            self.writer.add_text(f'Predictions/sample_{sample_index}', text, 0)

        except Exception as e:
            print(f"Warning: Failed to log predictions text: {e}")

    def log_histograms(self, model: Any, epoch: int):
        """
        Log weight and gradient histograms.

        Args:
            model: PyTorch model
            epoch: Current epoch number
        """
        if not self.enabled or self.writer is None:
            return

        if not self.config.get('histograms_enabled', False):
            return

        frequency = self.config.get('histograms_frequency', 100)
        if epoch % frequency != 0:
            return

        track_weights = self.config.get('histograms_track_weights', True)
        track_gradients = self.config.get('histograms_track_gradients', True)

        try:
            for name, param in model.named_parameters():
                if track_weights:
                    self.writer.add_histogram(f'Weights/{name}', param.data, epoch)

                if track_gradients and param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad.data, epoch)
        except Exception as e:
            print(f"Warning: Failed to log histograms: {e}")

    def log_model_graph(self, model: Any, input_tensor: Any):
        """
        Log model graph to TensorBoard.

        Args:
            model: PyTorch model
            input_tensor: Sample input tensor for tracing
        """
        if not self.enabled or self.writer is None:
            return

        if not self.config.get('model_graph_enabled', False):
            return

        try:
            self.writer.add_graph(model, input_tensor)
        except Exception as e:
            print(f"Warning: Failed to log model graph: {e}")

    def get_plots_dir(self) -> Optional[Path]:
        """Get the plots directory for saving figures."""
        if self.output_dir:
            plots_dir = self.output_dir / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            return plots_dir
        return None

    def get_checkpoints_dir(self) -> Optional[Path]:
        """Get the checkpoints directory for saving model weights."""
        if self.output_dir:
            checkpoints_dir = self.output_dir / 'checkpoints'
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            return checkpoints_dir
        return None

    def close(self):
        """Close the TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None
