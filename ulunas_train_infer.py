import os
import torch
import torch.nn as nn
# Enable TensorFloat32 for better performance
torch.set_float32_matmul_precision('high')
# Disable inductor autotune to avoid SM issues
torch._inductor.config.max_autotune = False
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import onnx
import onnxruntime as ort
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
import re
import math
from ulunas import ULUNAS
from dataset import DNSDataset
from loss import HybridLoss


# ======================================
# Global Configuration
# ======================================
# STFT parameters (aligned with agtcrn_train_infer.py)
HOP_LENGTH = 480          # Frame shift
N_FFT = HOP_LENGTH * 2    # FFT points = 960
WIN_LENGTH = N_FFT        # Window length = 960
FS = 48000                # Sampling rate

# Model configuration (aligned with agtcrn_train_infer.py)
ERB_SUBBAND_1 = 80        # ERB low subband
ERB_SUBBAND_2 = 41        # ERB high subband (80+41=121, odd number for symmetric conv/deconv)

# Dataset configuration
ROOT_DIR = '_datasets_fullband'  # Dataset root directory
SAMPLES_PER_EPOCH = 10000        # Samples per epoch

# Trainer configuration
EPOCHS = 1000               # Total epochs
BATCH_SIZE = 40             # Batch size
CLIP_GRAD_NORM = 50.0       # Gradient clipping threshold
SAVE_INTERVAL = 1           # Model save interval

# Learning rate configuration
MAX_LR = 1e-3               # Maximum learning rate
MIN_LR = 1e-6               # Minimum learning rate

# Export configuration
OPSET_VERSION = 14          # ONNX opset version

# Output directory configuration
OUTPUT_PREFIX = 'output_ulunas'  # Output directory prefix

# Loss record names (consistent with HybridLoss return order)
LOSS_A = 'A_Total'
LOSS_B = 'B_RI'
LOSS_C = 'C_Mag'
LOSS_D = 'D_Loudness'
LOSS_E = 'E_EnergyCeiling'
LOSS_F = 'F_ShortTime'
LOSS_G = 'G_SISNR'


def compute_stft(waveform, n_fft, hop_length, win_length, window, device):
    """Compute STFT from waveform."""
    complex_spec = torch.stft(
        waveform.squeeze(1),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True
    )
    return torch.view_as_real(complex_spec)


def compute_istft(complex_spec, n_fft, hop_length, win_length, window):
    """Compute inverse STFT from complex spectrogram."""
    # complex_spec is real tensor with shape (..., 2) representing real and imag parts
    complex_tensor = torch.view_as_complex(complex_spec.contiguous())
    return torch.istft(
        complex_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window
    )


def get_learning_rate(step, warmup_steps, cycle_steps, max_lr=1e-3, min_lr=1e-6):
    """Cosine learning rate schedule with warmup."""
    if step <= warmup_steps:
        lr = min_lr + (max_lr - min_lr) * (step / warmup_steps)
    else:
        step_after_warmup = step - warmup_steps
        lr = min_lr + (max_lr - min_lr) * 0.5 * (
            1 + math.cos(math.pi * min(step_after_warmup / cycle_steps, 1.0))
        )
    return max(lr, min_lr)


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory."""
    epoch_files = {}
    for f in os.listdir(checkpoint_dir):
        if f.endswith('.tar') and f.startswith('checkpoint_epoch_'):
            try:
                epoch_num = int(re.search(r'checkpoint_epoch_(\d+)\.tar', f).group(1))
                epoch_files[epoch_num] = f
            except:
                continue

    if epoch_files:
        max_epoch = max(epoch_files.keys())
        return os.path.join(checkpoint_dir, epoch_files[max_epoch]), max_epoch

    return None, 0


def load_wav(file_path, sr=48000):
    """Load waveform from file."""
    waveform, _ = librosa.load(file_path, sr=sr, mono=True)
    return torch.tensor(waveform).unsqueeze(0).unsqueeze(0)


def save_wav(file_path, waveform, sr=48000):
    """Save waveform to file."""
    sf.write(file_path, waveform.squeeze().cpu().numpy(), sr)


# ======================================
# Streaming Inference Classes
# ======================================

def get_kernel_size_t(module):
    """Extract time kernel size from conv module."""
    ks = module.kernel_size
    return ks[0] if isinstance(ks, (tuple, list)) else ks


class StreamingConvBuffer:
    """
    Strictly causal streaming convolution buffer.
    Offline: ZeroPad2d((0, 0, kt-1, 0)) pads kt-1 frames on the left (time dimension).
    Online: Maintain a buffer of size (B, C, kt-1, F), cat with current frame, conv, update buffer.
    """
    def __init__(self, kt, name):
        self.kt = kt
        self.name = name
        self.buffer = None

    def reset(self):
        self.buffer = None

    def init_buffer(self, B, C, F, device, dtype):
        # Buffer shape: (B, C, kt-1, F)
        # If kt=1, buffer is empty (no padding needed)
        if self.kt > 1:
            self.buffer = torch.zeros(B, C, self.kt - 1, F, device=device, dtype=dtype)
        else:
            self.buffer = torch.zeros(B, C, 0, F, device=device, dtype=dtype)

    def process(self, x):
        """
        Args:
            x: (B, C, 1, F) - single time frame
        Returns:
            x_cat: (B, C, kt, F) - concatenated with buffer for convolution
        """
        B, C, T, F = x.shape
        assert T == 1, f"Streaming expects T=1, got T={T}"

        if self.buffer is None:
            self.init_buffer(B, C, F, x.device, x.dtype)

        # If kt=1, no buffer needed, just return x
        if self.kt == 1:
            return x

        # Concatenate buffer with current frame: (B, C, kt, F)
        x_cat = torch.cat([self.buffer, x], dim=2)

        # Update buffer: keep last kt-1 frames
        self.buffer = x_cat[:, :, -(self.kt - 1):, :].clone()

        return x_cat


class StreamingGRUState:
    """Streaming GRU hidden state manager."""
    def __init__(self, name):
        self.name = name
        self.hidden = None

    def reset(self):
        self.hidden = None

    def init_hidden(self, num_layers, B, hidden_size, device, dtype, bidirectional=False):
        num_directions = 2 if bidirectional else 1
        self.hidden = torch.zeros(num_layers * num_directions, B, hidden_size, device=device, dtype=dtype)


class StreamULUNAS(nn.Module):
    """
    Strictly causal streaming ULUNAS model.
    Processes single frame at a time with explicit state management.
    Supports both PyTorch internal state and ONNX external state modes.

    State naming convention:
      - enc_{i}_conv_buffer: Encoder block i convolution buffer
      - enc_{i}_ta_gru_state: Encoder block i cTFA ta_gru hidden state
      - dpgrnn_{i}_inter_state: DPGRNN layer i inter_rnn hidden state
      - dec_{i}_conv_buffer: Decoder block i convolution buffer
      - dec_{i}_ta_gru_state: Decoder block i cTFA ta_gru hidden state
    """

    def __init__(self, model):
        super().__init__()
        self.n_fft = model.n_fft
        self.hop_len = model.hop_len
        self.win_len = model.win_len
        self.erb_subband_1 = model.erb.erb_subband_1
        self.erb_subband_2 = model.erb.erb_subband_2
        self.nfreqs = self.n_fft // 2 + 1

        # ERB modules
        self.erb = model.erb

        # Store original modules
        self.encoder = model.encoder
        self.dpgrnn = model.dpgrnn
        self.decoder = model.decoder

        # Initialize streaming buffers and states
        self._init_streaming_components()

    def _init_streaming_components(self):
        """Initialize all streaming buffers and states with explicit names."""
        # Encoder: each block has conv buffer and ta_gru state
        self.enc_conv_buffers = []
        self.enc_ta_gru_states = []

        for i, block in enumerate(self.encoder.en_convs):
            # Conv buffer for causal padding - different blocks have different structures
            block_type = type(block).__name__
            if block_type == 'XConvBlock':
                kt = get_kernel_size_t(block.ops[1])  # Conv kernel size
            elif block_type == 'XDWSBlock':
                kt = get_kernel_size_t(block.dconv[1])  # dconv kernel size
            elif block_type == 'XMBBlocks':
                kt = get_kernel_size_t(block.dconv[1])  # dconv kernel size
            else:
                raise ValueError(f"Unknown block type: {block_type}")

            # Only create buffer if kt > 1 (kernel size > 1 in time dimension)
            if kt > 1:
                buffer = StreamingConvBuffer(kt, f"enc_{i}_conv_buffer")
                self.enc_conv_buffers.append(buffer)
            else:
                self.enc_conv_buffers.append(None)  # No buffer needed for kt=1

            # ta_gru state for cTFA
            state = StreamingGRUState(f"enc_{i}_ta_gru_state")
            self.enc_ta_gru_states.append(state)

        # DPGRNN: each layer has inter_rnn state (intra_rnn is bidirectional, no state needed)
        self.dpgrnn_inter_states = []
        for i in range(len(self.dpgrnn)):
            state = StreamingGRUState(f"dpgrnn_{i}_inter_state")
            self.dpgrnn_inter_states.append(state)

        # Decoder: each block has conv buffer and ta_gru state
        self.dec_conv_buffers = []
        self.dec_ta_gru_states = []

        for i, block in enumerate(self.decoder.de_convs):
            block_type = type(block).__name__
            if block_type == 'XConvBlock':
                kt = get_kernel_size_t(block.ops[1])
            elif block_type == 'XDWSBlock':
                kt = get_kernel_size_t(block.dconv[1])
            elif block_type == 'XMBBlocks':
                kt = get_kernel_size_t(block.dconv[1])
            else:
                raise ValueError(f"Unknown block type: {block_type}")

            # Only create buffer if kt > 1
            if kt > 1:
                buffer = StreamingConvBuffer(kt, f"dec_{i}_conv_buffer")
                self.dec_conv_buffers.append(buffer)
            else:
                self.dec_conv_buffers.append(None)

            state = StreamingGRUState(f"dec_{i}_ta_gru_state")
            self.dec_ta_gru_states.append(state)

        # Encoder output cache for skip connections
        self.enc_outs_cache = []

    def reset_states(self):
        """Reset all streaming states for new session."""
        for buf in self.enc_conv_buffers:
            if buf is not None:
                buf.reset()
        for state in self.enc_ta_gru_states:
            state.reset()
        for state in self.dpgrnn_inter_states:
            state.reset()
        for buf in self.dec_conv_buffers:
            if buf is not None:
                buf.reset()
        for state in self.dec_ta_gru_states:
            state.reset()
        self.enc_outs_cache = []

    def _extract_states(self):
        """Extract all current states as a list (for ONNX export)."""
        states = []
        # Encoder states
        for buf in self.enc_conv_buffers:
            if buf is not None:
                states.append(buf.buffer)
        for state in self.enc_ta_gru_states:
            states.append(state.hidden)
        # DPGRNN states
        for state in self.dpgrnn_inter_states:
            states.append(state.hidden)
        # Decoder states
        for buf in self.dec_conv_buffers:
            if buf is not None:
                states.append(buf.buffer)
        for state in self.dec_ta_gru_states:
            states.append(state.hidden)
        return states

    def _load_states(self, states):
        """Load states from a list (for ONNX export)."""
        idx = 0
        # Encoder states
        for buf in self.enc_conv_buffers:
            if buf is not None:
                if states[idx] is not None:
                    buf.buffer = states[idx]
                idx += 1
        for gru_state in self.enc_ta_gru_states:
            if states[idx] is not None:
                gru_state.hidden = states[idx]
            idx += 1
        # DPGRNN states
        for gru_state in self.dpgrnn_inter_states:
            if states[idx] is not None:
                gru_state.hidden = states[idx]
            idx += 1
        # Decoder states
        for buf in self.dec_conv_buffers:
            if buf is not None:
                if states[idx] is not None:
                    buf.buffer = states[idx]
                idx += 1
        for gru_state in self.dec_ta_gru_states:
            if states[idx] is not None:
                gru_state.hidden = states[idx]
            idx += 1

    def _streaming_conv_block(self, block, x, conv_buffer, ta_gru_state):
        """
        Process one frame through XConvBlock/XDWSBlock/XMBBlocks with streaming.

        Args:
            block: XConvBlock or similar
            x: (B, C, 1, F) - single frame
            conv_buffer: StreamingConvBuffer
            ta_gru_state: StreamingGRUState

        Returns:
            out: (B, C_out, 1, F)
        """
        ops = block.ops
        zero_pad = ops[0]  # nn.ZeroPad2d (ignored in streaming, handled by buffer)
        conv = ops[1]      # Conv2d or ConvTranspose2d
        bn = ops[2]        # BatchNorm2d
        activation = ops[3]  # AffinePReLU or Identity
        ctfa = ops[4]      # cTFA
        shuffle = ops[5]   # Shuffle or Identity

        # Streaming convolution: concatenate buffer, conv, update buffer
        if conv_buffer is not None:
            x_cat = conv_buffer.process(x)  # (B, C, kt, F)
            x = conv(x_cat)  # (B, C_out, 1, F)
        else:
            # No buffer needed (kt=1)
            x = conv(x)
        x = bn(x)
        x = activation(x)

        # Streaming cTFA
        B, C, T, F = x.shape

        # Time attention: ta_gru is unidirectional, needs state
        zt = torch.mean(x.pow(2), dim=-1)  # (B, C, T)
        zt_t = zt.transpose(1, 2)  # (B, T, C) where T=1

        # Initialize hidden state if needed
        if ta_gru_state.hidden is None:
            # ta_gru has 1 layer, hidden_size = C*2
            ta_gru_state.init_hidden(1, B, C * 2, x.device, x.dtype, bidirectional=False)

        at, new_hidden = ctfa.ta_gru(zt_t, ta_gru_state.hidden)
        ta_gru_state.hidden = new_hidden
        at = ctfa.ta_fc(at).transpose(1, 2)  # (B, C, T)
        at = torch.sigmoid(at)

        # Frequency attention: FA operates on single frame, no state needed
        af = ctfa.fa(x)  # (B, T, F)
        af = torch.sigmoid(af)

        # Apply attention: at[..., None] * x * af[:, None, :, :]
        x = at[..., None] * x * af[:, None, :, :]

        x = shuffle(x)

        return x

    def _streaming_xdws_block(self, block, x, dconv_buffer, ta_gru_state):
        """
        Process one frame through XDWSBlock with streaming.
        XDWSBlock: pconv -> dconv (with buffer) -> ctfa
        """
        # Pointwise conv (no buffering needed)
        h = block.pconv(x)

        # Depthwise streaming conv
        dconv_ops = block.dconv
        conv = dconv_ops[1]
        bn = dconv_ops[2]
        activation = dconv_ops[3]
        ctfa = dconv_ops[4]

        if dconv_buffer is not None:
            x_cat = dconv_buffer.process(h)
            h = conv(x_cat)
        else:
            h = conv(h)
        h = bn(h)
        h = activation(h)

        # Streaming cTFA
        B, C, T, F = h.shape

        zt = torch.mean(h.pow(2), dim=-1)
        zt_t = zt.transpose(1, 2)

        if ta_gru_state.hidden is None:
            ta_gru_state.init_hidden(1, B, C * 2, h.device, h.dtype, bidirectional=False)

        at, new_hidden = ctfa.ta_gru(zt_t, ta_gru_state.hidden)
        ta_gru_state.hidden = new_hidden
        at = ctfa.ta_fc(at).transpose(1, 2)
        at = torch.sigmoid(at)

        af = ctfa.fa(h)
        af = torch.sigmoid(af)

        h = at[..., None] * h * af[:, None, :, :]

        return h

    def _streaming_xmb_block(self, block, x, dconv_buffer, ta_gru_state):
        """
        Process one frame through XMBBlocks with streaming.
        XMBBlocks: pconv1 -> dconv (with buffer) -> pconv2 -> residual -> shuffle
        """
        input_x = x

        # pconv1
        x = block.pconv1(x)

        # dconv with streaming buffer
        dconv_ops = block.dconv
        conv = dconv_ops[1]
        bn = dconv_ops[2]
        activation = dconv_ops[3]

        if dconv_buffer is not None:
            x_cat = dconv_buffer.process(x)
            x = conv(x_cat)
        else:
            x = conv(x)
        x = bn(x)
        x = activation(x)

        # pconv2 with cTFA
        x = block.pconv2[0](x)  # Conv2d
        x = block.pconv2[1](x)  # BatchNorm2d

        # cTFA
        ctfa = block.pconv2[2]
        B, C, T, F = x.shape

        zt = torch.mean(x.pow(2), dim=-1)
        zt_t = zt.transpose(1, 2)

        if ta_gru_state.hidden is None:
            ta_gru_state.init_hidden(1, B, C * 2, x.device, x.dtype, bidirectional=False)

        at, new_hidden = ctfa.ta_gru(zt_t, ta_gru_state.hidden)
        ta_gru_state.hidden = new_hidden
        at = ctfa.ta_fc(at).transpose(1, 2)
        at = torch.sigmoid(at)

        af = ctfa.fa(x)
        af = torch.sigmoid(af)

        x = at[..., None] * x * af[:, None, :, :]

        # Residual connection
        if x.shape == input_x.shape:
            x = x + input_x

        x = block.shuffle(x)

        return x

    def _streaming_encoder_block(self, block, x, conv_buffer, ta_gru_state):
        """Route to appropriate streaming block handler."""
        block_type = type(block).__name__

        if block_type == 'XConvBlock':
            return self._streaming_conv_block(block, x, conv_buffer, ta_gru_state)
        elif block_type == 'XDWSBlock':
            return self._streaming_xdws_block(block, x, conv_buffer, ta_gru_state)
        elif block_type == 'XMBBlocks':
            return self._streaming_xmb_block(block, x, conv_buffer, ta_gru_state)
        else:
            raise ValueError(f"Unknown block type: {block_type}")

    def _streaming_dpgrnn(self, dpgrnn, x, inter_state):
        """
        Process one frame through DPGRNN with streaming.

        Args:
            dpgrnn: DPGRNN module
            x: (B, C, 1, F) - single frame
            inter_state: StreamingGRUState for inter_rnn

        Returns:
            out: (B, C, 1, F)
        """
        # x: (B, C, T, F) where T=1
        B, C, T, F_dim = x.shape

        # Intra RNN: bidirectional on frequency axis, no state needed
        x_perm = x.permute(0, 2, 3, 1)  # (B, T, F, C)
        intra_x = x_perm.reshape(B * T, F_dim, C)  # (B*T, F, C)
        intra_x = dpgrnn.intra_rnn(intra_x)[0]  # (B*T, F, C)
        intra_x = dpgrnn.intra_fc(intra_x)  # (B*T, F, C)
        intra_x = intra_x.reshape(B, T, F_dim, C)  # (B, T, F, C)
        intra_x = dpgrnn.intra_ln(intra_x)
        intra_out = torch.add(x_perm, intra_x)  # (B, T, F, C)

        # Inter RNN: unidirectional on time axis, needs state
        x_inter = intra_out.permute(0, 2, 1, 3)  # (B, F, T, C)
        x_inter = x_inter.reshape(B * F_dim, T, C)  # (B*F, T, C)

        # Initialize hidden state if needed
        if inter_state.hidden is None:
            # inter_rnn is GRNN with hidden_size
            inter_state.init_hidden(1, B * F_dim, dpgrnn.inter_rnn.hidden_size,
                                   x.device, x.dtype, bidirectional=False)

        inter_x, new_hidden = dpgrnn.inter_rnn(x_inter, inter_state.hidden)
        inter_state.hidden = new_hidden

        inter_x = dpgrnn.inter_fc(inter_x)  # (B*F, T, C)
        inter_x = inter_x.reshape(B, F_dim, T, C)  # (B, F, T, C)
        inter_x = inter_x.permute(0, 2, 1, 3)  # (B, T, F, C)
        inter_x = dpgrnn.inter_ln(inter_x)
        inter_out = torch.add(intra_out, inter_x)  # (B, T, F, C)

        # Permute back to (B, C, T, F)
        out = inter_out.permute(0, 3, 1, 2)  # (B, C, T, F)

        return out

    def forward(self, spec_frame, *external_states):
        """
        Process a single frame with explicit state management.

        Args:
            spec_frame: (B, F, 1, 2) - single frame spectrogram
            *external_states: Optional external states for ONNX export
                            If provided, states are loaded from here and returned

        Returns:
            enhanced_frame: (B, F, 1, 2) - enhanced frame
            *new_states: New states (if external_states provided)
        """
        use_external_states = len(external_states) > 0

        if use_external_states:
            self._load_states(list(external_states))

        # Convert to (B, 2, T, F) format
        spec = spec_frame.permute(0, 3, 2, 1)  # (B, 2, 1, F)

        # Compute magnitude feature
        feat = torch.log10(torch.norm(spec, dim=1, keepdim=True).clamp(1e-12))

        # ERB band merging
        feat = self.erb.bm(feat)  # (B, 1, 1, F_erb)

        # Encoder with streaming
        en_outs = []
        x = feat

        for i, block in enumerate(self.encoder.en_convs):
            x = self._streaming_encoder_block(
                block, x,
                self.enc_conv_buffers[i],
                self.enc_ta_gru_states[i]
            )
            en_outs.append(x)

        # DPGRNN with streaming
        for i, dpgrnn_layer in enumerate(self.dpgrnn):
            x = self._streaming_dpgrnn(
                dpgrnn_layer, x,
                self.dpgrnn_inter_states[i]
            )

        # Decoder with streaming and skip connections
        n_blocks = len(self.decoder.de_convs)
        for i in range(n_blocks):
            skip = en_outs[n_blocks - i - 1]
            block = self.decoder.de_convs[i]

            # Add skip connection then process
            x = self._streaming_encoder_block(
                block, x + skip,
                self.dec_conv_buffers[i],
                self.dec_ta_gru_states[i]
            )

        # Apply sigmoid mask
        m_feat = torch.sigmoid(x)

        # ERB band splitting
        m = self.erb.bs(m_feat)

        # Apply mask
        spec_enh = spec * m

        # Convert back to (B, F, T, 2) format
        spec_enh = spec_enh.permute(0, 3, 2, 1)  # (B, F, 1, 2)

        if use_external_states:
            new_states = self._extract_states()
            return (spec_enh, *new_states)

        return spec_enh


def infer_with_pytorch(model, wav_path, output_path, device, use_streaming=False):
    """
    Inference using PyTorch model.

    Args:
        model: ULUNAS model
        wav_path: Input wav file path
        output_path: Output wav file path
        device: torch device
        use_streaming: Whether to use streaming inference
    """
    print(f"\n=== PyTorch Inference: {wav_path} ===")

    waveform = load_wav(wav_path, sr=FS).to(device)
    n_samples = waveform.shape[-1]

    window = torch.hann_window(WIN_LENGTH).to(device)
    spec = compute_stft(waveform, N_FFT, HOP_LENGTH, WIN_LENGTH, window, device)
    # spec: (B, F, T, 2)

    model.eval()
    with torch.no_grad():
        if use_streaming:
            # Streaming inference - frame by frame with explicit state management
            stream_model = StreamULUNAS(model).to(device).eval()
            stream_model.reset_states()  # Reset all streaming states
            FRAMES = spec.shape[2]
            enhanced_spec = torch.zeros_like(spec)

            for t in range(FRAMES):
                current_frame = spec[:, :, t:t + 1, :]
                frame_output = stream_model(current_frame)
                enhanced_spec[:, :, t:t + 1, :] = frame_output
        else:
            # Offline inference - full sequence
            # ULUNAS expects (B, n_samples) input
            enhanced_waveform = model(waveform.squeeze(1))
            # Convert back to spectrogram for consistent output format
            enhanced_spec = compute_stft(
                enhanced_waveform.unsqueeze(1),
                N_FFT, HOP_LENGTH, WIN_LENGTH, window, device
            )

    if use_streaming:
        # enhanced_spec is (B, F, T, 2) real tensor
        enhanced_waveform = compute_istft(
            enhanced_spec, N_FFT, HOP_LENGTH, WIN_LENGTH, window
        ).unsqueeze(1)
    else:
        enhanced_waveform = enhanced_waveform.unsqueeze(1)

    # Pad to original length
    if enhanced_waveform.shape[-1] < n_samples:
        enhanced_waveform = torch.nn.functional.pad(
            enhanced_waveform, (0, n_samples - enhanced_waveform.shape[-1])
        )
    else:
        enhanced_waveform = enhanced_waveform[..., :n_samples]

    save_wav(output_path, enhanced_waveform, sr=FS)
    print(f"PyTorch inference saved to: {output_path}")

    return enhanced_waveform


def convert_model_to_onnx(model, onnx_path, device):
    """
    Convert streaming ULUNAS model to ONNX format with explicit state I/O.
    This exports a strictly causal streaming model that processes one frame at a time
    with state passing between frames.

    Args:
        model: ULUNAS model
        onnx_path: Output ONNX file path
        device: torch device
    """
    model.eval()

    # Create streaming model for export
    stream_model = StreamULUNAS(model).to(device).eval()
    stream_model.reset_states()

    # Dummy input: single frame spectrogram (B, F, 1, 2)
    FREQ = N_FFT // 2 + 1
    dummy_input = torch.randn(1, FREQ, 1, 2).to(device)

    try:
        # Export with state I/O for true streaming inference
        # First, run a forward pass to initialize all states
        with torch.no_grad():
            _ = stream_model(dummy_input)  # Initialize internal states

        # Now extract initialized states
        initial_states = stream_model._extract_states()

        # Create dummy state tensors with correct shapes
        dummy_states = []
        input_names = ['spec']
        output_names = ['enhanced_spec']
        dynamic_axes = {
            'spec': {0: 'batch'},
            'enhanced_spec': {0: 'batch'}
        }

        for i, state in enumerate(initial_states):
            state_name = f'state_{i}'
            new_state_name = f'new_state_{i}'
            input_names.append(state_name)
            output_names.append(new_state_name)

            # State should now be initialized with correct shape
            if state is not None:
                dummy_states.append(state)
                # Add dynamic axes for batch dimension
                dynamic_axes[state_name] = {0: 'batch'}
                dynamic_axes[new_state_name] = {0: 'batch'}
            else:
                # This shouldn't happen after forward pass, but handle gracefully
                raise ValueError(f"State {i} is None after forward pass. Check state initialization.")

        dummy_inputs = (dummy_input, *dummy_states)

        torch.onnx.export(
            stream_model,
            dummy_inputs,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
            dynamo=False,
            dynamic_axes=dynamic_axes
        )
        print(f"Streaming ONNX model exported: {onnx_path}")
        print(f"  - Input: spec {dummy_input.shape}")
        print(f"  - States: {len(dummy_states)} state tensors")
        print(f"  - Output: enhanced_spec + {len(dummy_states)} new states")

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed")

        try:
            from onnxsim import simplify
            print("Simplifying ONNX model...")
            model_simp, check = simplify(onnx_model)
            assert check, "Simplified ONNX model validation failed"
            onnx.save(model_simp, onnx_path)
            print("ONNX model simplified")
        except ImportError:
            print("onnxsim not installed, skipping simplification")
        except Exception as e:
            print(f"ONNX simplification failed: {e}")

        return True
    except Exception as e:
        print(f"ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def infer_with_onnx(onnx_path, wav_path, output_path, use_states=True):
    """
    Inference using ONNX model.

    Args:
        onnx_path: ONNX model path
        wav_path: Input wav file path
        output_path: Output wav file path
        use_states: If True, use explicit state management for true streaming
                   (default True, as stateless mode cannot maintain frame continuity)
    """
    print(f"\n=== ONNX Inference: {wav_path} ===")
    print(f"Mode: {'Streaming with states' if use_states else 'Frame-by-frame (WARNING: states not preserved between frames)'}")

    waveform = load_wav(wav_path, sr=FS)
    n_samples = waveform.shape[-1]
    window = torch.hann_window(WIN_LENGTH)
    spec = compute_stft(waveform, N_FFT, HOP_LENGTH, WIN_LENGTH, window, 'cpu')

    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # Get input/output info
    input_names = [inp.name for inp in ort_session.get_inputs()]
    output_names = [out.name for out in ort_session.get_outputs()]

    # Check if model supports states
    model_has_states = len(input_names) > 1

    if use_states and not model_has_states:
        print("WARNING: Model does not support states, falling back to stateless mode")
        print("         (Results may be incorrect for streaming inference)")
        use_states = False

    # Process frame by frame
    FRAMES = spec.shape[2]
    enhanced_spec = np.zeros_like(spec.numpy())

    # Initialize states if using stateful inference
    states = None
    if use_states:
        # Initialize states to zeros
        states = []
        for inp in ort_session.get_inputs()[1:]:  # Skip 'spec' input
            # Handle dynamic axes (None or string dimensions)
            state_shape = []
            for d in inp.shape:
                if d is None or isinstance(d, str):
                    state_shape.append(1)  # Use batch size 1 for initialization
                else:
                    state_shape.append(int(d))
            states.append(np.zeros(state_shape, dtype=np.float32))

    for t in range(FRAMES):
        current_frame = spec[:, :, t:t + 1, :].numpy()

        if use_states:
            # Stateful inference: pass states as inputs
            inputs = {'spec': current_frame}
            for name, state in zip(input_names[1:], states):
                inputs[name] = state

            outputs = ort_session.run(None, inputs)
            enhanced_spec[:, :, t:t + 1, :] = outputs[0]

            # Update states for next frame
            states = list(outputs[1:])
        else:
            # Stateless inference (WARNING: states not preserved between frames!)
            outputs = ort_session.run(None, {'spec': current_frame})
            enhanced_spec[:, :, t:t + 1, :] = outputs[0]

    # Convert back to waveform
    enhanced_spec_complex = torch.view_as_complex(torch.tensor(enhanced_spec))
    enhanced_waveform = torch.istft(
        enhanced_spec_complex,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window
    )

    # Pad or trim to match original length
    if enhanced_waveform.shape[-1] < n_samples:
        enhanced_waveform = torch.nn.functional.pad(
            enhanced_waveform, (0, n_samples - enhanced_waveform.shape[-1])
        )
    else:
        enhanced_waveform = enhanced_waveform[..., :n_samples]

    save_wav(output_path, enhanced_waveform.unsqueeze(0).unsqueeze(0), sr=FS)
    print(f"ONNX inference saved to: {output_path}")

    return enhanced_waveform.unsqueeze(0)


def compare_results(pytorch_result, onnx_result):
    """Compare PyTorch and ONNX inference results."""
    print("\n=== Comparing PyTorch and ONNX Results ===")

    pytorch_result = pytorch_result.cpu()
    onnx_result = onnx_result.cpu()

    # Handle shape mismatch by aligning lengths
    if pytorch_result.shape != onnx_result.shape:
        print(f"Shape mismatch: PyTorch={pytorch_result.shape}, ONNX={onnx_result.shape}")
        # Align to minimum length
        min_len = min(pytorch_result.shape[-1], onnx_result.shape[-1])
        pytorch_result = pytorch_result[..., :min_len]
        onnx_result = onnx_result[..., :min_len]
        print(f"Aligned to length: {min_len}")

    # 1. Basic statistics
    print("\n1. Basic Statistics:")
    print(f"   PyTorch mean: {torch.mean(pytorch_result).item():.6f}")
    print(f"   ONNX mean: {torch.mean(onnx_result).item():.6f}")
    print(f"   PyTorch std: {torch.std(pytorch_result).item():.6f}")
    print(f"   ONNX std: {torch.std(onnx_result).item():.6f}")
    print(f"   PyTorch max: {torch.max(pytorch_result).item():.6f}")
    print(f"   ONNX max: {torch.max(onnx_result).item():.6f}")
    print(f"   PyTorch min: {torch.min(pytorch_result).item():.6f}")
    print(f"   ONNX min: {torch.min(onnx_result).item():.6f}")

    # 2. Error metrics
    mse_tensor = torch.mean((pytorch_result - onnx_result) ** 2)
    mse = mse_tensor.item()
    mae = torch.mean(torch.abs(pytorch_result - onnx_result)).item()
    rmse = torch.sqrt(mse_tensor).item()
    print("\n2. Error Metrics:")
    print(f"   MSE: {mse:.10f}")
    print(f"   MAE: {mae:.10f}")
    print(f"   RMSE: {rmse:.10f}")

    # 3. Signal quality
    print("\n3. Signal Quality:")
    signal_power = torch.mean(pytorch_result ** 2)
    noise_power = torch.mean((pytorch_result - onnx_result) ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-12)).item()
    print(f"   SNR: {snr:.6f} dB")

    # 4. Correlation
    corr_matrix = torch.corrcoef(
        torch.stack([pytorch_result.squeeze(), onnx_result.squeeze()])
    )
    pearson_corr = corr_matrix[0, 1].item()
    print("\n4. Correlation:")
    print(f"   Pearson Correlation: {pearson_corr:.10f}")

    # 5. Waveform similarity
    print("\n5. Waveform Similarity:")
    cosine_sim = torch.nn.functional.cosine_similarity(
        pytorch_result.squeeze(), onnx_result.squeeze(), dim=0
    ).item()
    print(f"   Cosine Similarity: {cosine_sim:.10f}")

    # 6. Difference distribution
    print("\n6. Difference Distribution:")
    max_diff = torch.max(torch.abs(pytorch_result - onnx_result)).item()
    min_diff = torch.min(torch.abs(pytorch_result - onnx_result)).item()
    median_diff = torch.median(torch.abs(pytorch_result - onnx_result)).item()
    print(f"   Max absolute diff: {max_diff:.10f}")
    print(f"   Min absolute diff: {min_diff:.10f}")
    print(f"   Median absolute diff: {median_diff:.10f}")

    # 7. Error ratio
    print("\n7. Error Ratio:")
    total_energy = torch.sum(pytorch_result ** 2).item()
    error_energy = torch.sum((pytorch_result - onnx_result) ** 2).item()
    error_ratio = error_energy / total_energy
    print(f"   Error energy ratio: {error_ratio:.10f} ({error_ratio*100:.8f}%)")

    # 8. Frequency domain analysis
    print("\n8. Frequency Domain Analysis:")
    window = torch.hann_window(WIN_LENGTH)
    pytorch_stft = torch.stft(
        pytorch_result.squeeze(), n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, window=window, return_complex=True
    )
    onnx_stft = torch.stft(
        onnx_result.squeeze(), n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, window=window, return_complex=True
    )
    pytorch_mag = torch.abs(pytorch_stft)
    onnx_mag = torch.abs(onnx_stft)
    pytorch_phase = torch.angle(pytorch_stft)
    onnx_phase = torch.angle(onnx_stft)
    mag_mse = torch.mean((pytorch_mag - onnx_mag) ** 2).item()
    phase_mse = torch.mean((pytorch_phase - onnx_phase) ** 2).item()
    print(f"   Magnitude MSE: {mag_mse:.10f}")
    print(f"   Phase MSE: {phase_mse:.10f}")

    # 9. Consistency check
    print("\n9. Consistency:")
    if mse < 1e-6 and snr > 60 and pearson_corr > 0.9999:
        print("   High consistency!")
    elif mse < 1e-5 and snr > 40 and pearson_corr > 0.999:
        print("   Good consistency!")
    elif mse < 1e-4 and pearson_corr > 0.99:
        print("   Basic consistency")
    else:
        print("   Significant difference detected!")

    # 10. Summary report
    print("\n10. Summary Report:")
    score = min(100, int((1 - min(mse, 1e-3) / 1e-3) * 100))
    passed = mse < 1e-6 and snr > 60 and pearson_corr > 0.9999
    print(f"   Consistency Score: {score}/100")
    print(f"   Verification Passed: {'Yes' if passed else 'No'}")
    print(f"   Recommendation: {'Deploy ONNX model' if passed else 'Check conversion parameters'}")
    focus = 'Phase difference' if phase_mse > mag_mse else 'Magnitude difference' if mag_mse > 1e-6 else 'Good overall consistency'
    print(f"   Focus Area: {focus}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='ULUNAS Training and Inference')
    parser.add_argument('--test_wav', type=str, default='merged_noisy_files.wav',
                        help='Test wav file path')
    parser.add_argument('--bf16', action='store_true', default=True,
                        help='Use bf16 mixed precision training (default: True)')
    parser.add_argument('--no-bf16', action='store_true',
                        help='Disable bf16 mixed precision training')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'infer', 'stream'],
                        help='Mode: train, infer (offline), or stream')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path for inference')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Enable bf16 mixed precision (default)
    use_bf16 = not args.no_bf16 and torch.cuda.is_available()
    if use_bf16:
        print("Using bf16 mixed precision training")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Output directories
    checkpoint_dir = f'{OUTPUT_PREFIX}/checkpoints'
    onnx_dir = f'{OUTPUT_PREFIX}/onnx_models'
    output_dir = f'{OUTPUT_PREFIX}/results_wav'
    tb_dir = f'{OUTPUT_PREFIX}/logs'

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(onnx_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    # Initialize STFT window
    window = torch.hann_window(WIN_LENGTH).to(device)

    # Inference mode
    if args.mode in ['infer', 'stream']:
        if args.checkpoint is None:
            latest_checkpoint, _ = find_latest_checkpoint(checkpoint_dir)
            if latest_checkpoint is None:
                print("No checkpoint found for inference")
                return
            args.checkpoint = latest_checkpoint

        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)

        model = ULUNAS(
            n_fft=N_FFT,
            hop_len=HOP_LENGTH,
            win_len=WIN_LENGTH,
            erb_low=ERB_SUBBAND_1,
            erb_high=ERB_SUBBAND_2
        ).to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        if args.test_wav and os.path.exists(args.test_wav):
            use_streaming = (args.mode == 'stream')
            output_path = os.path.join(
                output_dir,
                f'{args.mode}_{os.path.basename(args.test_wav)}'
            )
            infer_with_pytorch(model, args.test_wav, output_path, device,
                              use_streaming=use_streaming)
        return

    # Training mode
    latest_checkpoint, resume_epoch = find_latest_checkpoint(checkpoint_dir)

    start_epoch = 0
    global_step = 0
    resume_training = False

    if latest_checkpoint:
        try:
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            start_epoch = checkpoint.get('epoch', 0) + 1
            global_step = checkpoint.get('global_step', 0)
            resume_training = True
            print(f"Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    writer = SummaryWriter(log_dir=tb_dir, purge_step=global_step if resume_training else None)

    def collate_fn(batch):
        max_length = max([x[0].shape[1] for x in batch])
        padded_noisy = torch.zeros(len(batch), 1, max_length)
        padded_clean = torch.zeros(len(batch), 1, max_length)

        for i, (noisy, clean) in enumerate(batch):
            length = noisy.shape[1]
            padded_noisy[i, :, :length] = noisy
            padded_clean[i, :, :length] = clean

        return padded_noisy, padded_clean

    full_train_dataset = DNSDataset(root_dir=ROOT_DIR, train=True)
    steps_per_epoch = SAMPLES_PER_EPOCH // BATCH_SIZE

    warmup_steps = steps_per_epoch * 10
    cycle_steps = steps_per_epoch * EPOCHS

    val_dataset = DNSDataset(root_dir=ROOT_DIR, train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )

    model = ULUNAS(
        n_fft=N_FFT,
        hop_len=HOP_LENGTH,
        win_len=WIN_LENGTH,
        erb_low=ERB_SUBBAND_1,
        erb_high=ERB_SUBBAND_2
    ).to(device).train()

    # Compile model for faster training
    print("Compiling model with torch.compile...")
    model = torch.compile(model)

    loss_func = HybridLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=1e-2)

    # Initialize bf16 GradScaler
    scaler = torch.amp.GradScaler('cuda', enabled=use_bf16)

    if resume_training:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        # Restore learning rate schedule
        for step in range(global_step):
            lr = get_learning_rate(step, warmup_steps, cycle_steps, MAX_LR, MIN_LR)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    total_epochs = EPOCHS

    for epoch in range(start_epoch, total_epochs + 1):
        # Create subset sampler for this epoch
        indices = np.random.choice(len(full_train_dataset), SAMPLES_PER_EPOCH, replace=False)
        sampler = SubsetRandomSampler(indices)

        train_loader = DataLoader(
            full_train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )

        model.train()
        epoch_losses = {k: 0.0 for k in ['A', 'B', 'C', 'D', 'E', 'F', 'G']}

        pbar = tqdm(train_loader, total=len(train_loader),
                   desc=f'Epoch {epoch}/{total_epochs}', ncols=70)

        for noisy_waveform, clean_waveform in pbar:
            # Update learning rate
            lr = get_learning_rate(global_step, warmup_steps, cycle_steps, MAX_LR, MIN_LR)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            writer.add_scalar('1_All/Learning_Rate', lr, global_step)

            noisy_waveform = noisy_waveform.to(device)
            clean_waveform = clean_waveform.to(device)

            # Forward pass with bf16
            with torch.amp.autocast('cuda', enabled=use_bf16, dtype=torch.bfloat16):
                # ULUNAS expects (B, n_samples) input
                enhanced_waveform = model(noisy_waveform.squeeze(1))

                # Compute STFT for loss calculation
                enhanced_spec = compute_stft(
                    enhanced_waveform.unsqueeze(1),
                    N_FFT, HOP_LENGTH, WIN_LENGTH, window, device
                )
                clean_spec = compute_stft(
                    clean_waveform,
                    N_FFT, HOP_LENGTH, WIN_LENGTH, window, device
                )

                losses = loss_func(enhanced_spec, clean_spec)
                total_loss = losses[0]

            # Backward pass
            optimizer.zero_grad()
            if use_bf16:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            # Gradient clipping
            if CLIP_GRAD_NORM > 0:
                if use_bf16:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=CLIP_GRAD_NORM
                )
                writer.add_scalar('1_All/Gradient_Norm', grad_norm.item(), global_step)

            # Optimizer step
            if use_bf16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # Record losses
            loss_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
            for i, name in enumerate(loss_names):
                loss_val = losses[i].item()
                epoch_losses[name] += loss_val
                writer.add_scalar(f'2_Train/{name}', loss_val, global_step)

            global_step += 1

        # Epoch summary
        avg_train_losses = {k: v / len(train_loader) for k, v in epoch_losses.items()}
        for k, v in avg_train_losses.items():
            writer.add_scalar(f'3_Epoch_Train/{k}', v, epoch)

        # Validation
        model.eval()
        val_losses = {k: 0.0 for k in ['A', 'B', 'C', 'D', 'E', 'F', 'G']}

        with torch.no_grad():
            pbar_val = tqdm(val_loader, total=len(val_loader), desc='Validation', ncols=70)
            for noisy_waveform, clean_waveform in pbar_val:
                noisy_waveform = noisy_waveform.to(device)
                clean_waveform = clean_waveform.to(device)

                with torch.amp.autocast('cuda', enabled=use_bf16, dtype=torch.bfloat16):
                    enhanced_waveform = model(noisy_waveform.squeeze(1))
                    enhanced_spec = compute_stft(
                        enhanced_waveform.unsqueeze(1),
                        N_FFT, HOP_LENGTH, WIN_LENGTH, window, device
                    )
                    clean_spec = compute_stft(
                        clean_waveform,
                        N_FFT, HOP_LENGTH, WIN_LENGTH, window, device
                    )
                    losses = loss_func(enhanced_spec, clean_spec)

                for i, name in enumerate(loss_names):
                    val_losses[name] += losses[i].item()

        avg_val_losses = {k: v / len(val_loader) for k, v in val_losses.items()}
        for k, v in avg_val_losses.items():
            writer.add_scalar(f'4_Epoch_Val/{k}', v, epoch)

        # Save checkpoint
        checkpoint_data = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
            'train_loss': avg_train_losses['A'],
            'val_loss': avg_val_losses['A']
        }
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.tar')
        torch.save(checkpoint_data, checkpoint_path)

        # Export streaming ONNX model with state I/O
        onnx_path = os.path.join(onnx_dir, f'ulunas_epoch_{epoch}.onnx')
        convert_model_to_onnx(model, onnx_path, device)

        # Test inference
        if args.test_wav and os.path.exists(args.test_wav):
            # PyTorch offline inference
            pytorch_output_path = os.path.join(
                output_dir,
                f'torch_{epoch}_{os.path.basename(args.test_wav)}'
            )
            pytorch_result = infer_with_pytorch(model, args.test_wav, pytorch_output_path, device,
                              use_streaming=False)

            # ONNX streaming inference with states
            # onnx_output_path = os.path.join(output_dir, f'onnx_{epoch}_{os.path.basename(args.test_wav)}')
            # onnx_result = infer_with_onnx(onnx_path, args.test_wav, onnx_output_path, use_states=True)

            # # Compare PyTorch offline and ONNX streaming results
            # compare_results(pytorch_result, onnx_result)

        if epoch % 10 == 0:
            writer.flush()

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, 'final_model.tar')
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': total_epochs,
        'global_step': global_step
    }, final_model_path)

    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    main()
