#!/usr/bin/env python3
"""
PS4 — Target Speaker Extraction Inference Script
=================================================

Self-contained inference script for the PS4 TSE model.
No external dependencies beyond torch, torchaudio, and numpy.

Usage:
    # Basic inference
    python inference.py \\
        --checkpoint checkpoint_epoch037.pt \\
        --mix mix.wav \\
        --enroll target_speaker.wav \\
        --output result.wav

    # Use GPU
    python inference.py \\
        --checkpoint checkpoint_epoch037.pt \\
        --mix mix.wav \\
        --enroll target.wav \\
        --output result.wav \\
        --device cuda

    # Batch mode (process a directory of mixtures with one enrollment per file)
    python inference.py \\
        --checkpoint checkpoint_epoch037.pt \\
        --mix-dir ./mixtures/ \\
        --enroll-dir ./enrollments/ \\
        --output-dir ./results/ \\
        --device cuda

    # List available CUDA devices
    python inference.py --list-devices
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ============================================================================
# Helper: LinearLayer (used by SpeakerFuseLayer)
# ============================================================================

class LinearLayer(nn.Module):
    """Simple linear layer with a dummy second argument for compatibility."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x, dummy: Optional[torch.Tensor] = None):
        return self.linear(x)


# ============================================================================
# Speaker helper modules
# ============================================================================

class PreEmphasis(nn.Module):
    """Pre-emphasis filter: y(t) = x(t) - coef * x(t-1)."""

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            "flipped_filter",
            torch.FloatTensor([-self.coef, 1.0]).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), "reflect")
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class SpeakerTransform(nn.Module):
    """Transform speaker embeddings through a series of 1x1 conv layers."""

    def __init__(self, embed_dim=256, num_layers=3, hid_dim=128):
        super().__init__()
        layers = []
        layers.append(nn.Conv1d(embed_dim, hid_dim, 1))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv1d(hid_dim, hid_dim, 1))
            layers.append(nn.Tanh())
        layers.append(nn.Conv1d(hid_dim, embed_dim, 1))
        self.transforms = nn.Sequential(*layers)

    def forward(self, x):
        if len(x.size()) == 2:
            return self.transforms(x.unsqueeze(-1)).squeeze(-1)
        return self.transforms(x)


class SpeakerFuseLayer(nn.Module):
    """Fuse speaker embedding with audio features via various fusion strategies."""

    def __init__(self, embed_dim=256, feat_dim=512, fuse_type="concat"):
        super().__init__()
        assert fuse_type in ["concat", "additive", "multiply", "FiLM", "None"]
        self.fuse_type = fuse_type
        if fuse_type == "concat":
            self.fc = LinearLayer(embed_dim + feat_dim, feat_dim)
        elif fuse_type in ("additive", "multiply"):
            self.fc = LinearLayer(embed_dim, feat_dim)
        elif fuse_type == "FiLM":
            raise NotImplementedError("FiLM not supported in this standalone script")
        else:
            raise ValueError(f"Fuse type not defined: {fuse_type}")

    def forward(self, x, embed):
        if self.fuse_type == "concat":
            if len(x.size()) == 3:
                embed_t = embed.expand(-1, -1, x.size(2))
                y = torch.cat([x, embed_t], 1)
                y = torch.transpose(y, 1, 2)
                x = torch.transpose(self.fc(y), 1, 2)
            else:
                embed_t = embed.expand(-1, x.size(1), -1, x.size(3))
                y = torch.cat([x, embed_t], 2)
                y = torch.transpose(y, 2, 3)
                x = torch.transpose(self.fc(y), 2, 3).contiguous()
        elif self.fuse_type == "additive":
            if len(x.size()) == 3:
                embed_t = embed.expand(-1, -1, x.size(2))
                embed_t = torch.transpose(embed_t, 1, 2)
                x = x + torch.transpose(self.fc(embed_t), 1, 2)
            else:
                embed_t = embed.expand(-1, x.size(1), -1, x.size(3))
                embed_t = torch.transpose(embed_t, 2, 3)
                x = x + torch.transpose(self.fc(embed_t), 2, 3)
        elif self.fuse_type == "multiply":
            if len(x.size()) == 3:
                embed_t = embed.expand(-1, -1, x.size(2))
                embed_t = torch.transpose(embed_t, 1, 2)
                x = x * torch.transpose(self.fc(embed_t), 1, 2)
            else:
                embed_t = embed.expand(-1, x.size(1), -1, x.size(3))
                embed_t = torch.transpose(embed_t, 2, 3)
                x = x * torch.transpose(self.fc(embed_t), 2, 3)
        else:
            embed = embed.squeeze(-1)
            x = self.fc(embed, x)
        return x


# ============================================================================
# ECAPA-TDNN Speaker Encoder (for joint speaker embedding extraction)
# ============================================================================

class Conv1dReluBn(nn.Module):
    """Conv1d + BatchNorm1d + ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class Res2Conv1dReluBn(nn.Module):
    """Res2Conv1d + BatchNorm1d + ReLU."""

    def __init__(self, channels, kernel_size=1, stride=1, padding=0,
                 dilation=1, bias=True, scale=4):
        super().__init__()
        assert channels % scale == 0, f"{channels} % {scale} != 0"
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(self.nums):
            self.convs.append(
                nn.Conv1d(self.width, self.width, kernel_size,
                          stride, padding, dilation, bias=bias))
            self.bns.append(nn.BatchNorm1d(self.width))

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        sp = spx[0]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if i >= 1:
                sp = sp + spx[i]
            sp = conv(sp)
            sp = bn(F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        return torch.cat(out, dim=1)


class SE_Connect(nn.Module):
    """Squeeze-Excitation block for 1D."""

    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        return x * out.unsqueeze(2)


class SE_Res2Block(nn.Module):
    """SE-Res2Block of the ECAPA-TDNN architecture."""

    def __init__(self, channels, kernel_size, stride, padding, dilation, scale):
        super().__init__()
        self.se_res2block = nn.Sequential(
            Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
            Res2Conv1dReluBn(channels, kernel_size, stride, padding, dilation, scale=scale),
            Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
            SE_Connect(channels),
        )

    def forward(self, x):
        return x + self.se_res2block(x)


class ASTP(nn.Module):
    """Attentive statistics pooling: first used in ECAPA-TDNN."""

    def __init__(self, in_dim, bottleneck_dim=128, global_context_att=False, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.global_context_att = global_context_att
        if global_context_att:
            self.linear1 = nn.Conv1d(in_dim * 3, bottleneck_dim, kernel_size=1)
        else:
            self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        assert len(x.shape) == 3

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-7).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        alpha = torch.tanh(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-7))
        return torch.cat([mean, std], dim=1)

    def get_out_dim(self):
        return 2 * self.in_dim


class ECAPA_TDNN(nn.Module):
    """ECAPA-TDNN speaker encoder."""

    def __init__(self, channels=512, feat_dim=80, embed_dim=192,
                 pooling_func="ASTP", global_context_att=False, emb_bn=False):
        super().__init__()
        self.layer1 = Conv1dReluBn(feat_dim, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        self.layer4 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)

        cat_channels = channels * 3
        out_channels = 512 * 3
        self.conv = nn.Conv1d(cat_channels, out_channels, kernel_size=1)
        self.pool = ASTP(in_dim=out_channels, global_context_att=global_context_att)
        self.pool_out_dim = self.pool.get_out_dim()
        self.bn = nn.BatchNorm1d(self.pool_out_dim)
        self.linear = nn.Linear(self.pool_out_dim, embed_dim)
        self.emb_bn = emb_bn
        self.bn2 = nn.BatchNorm1d(embed_dim) if emb_bn else nn.Identity()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, T, F) -> (B, F, T)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = torch.cat([out2, out3, out4], dim=1)
        out = self.conv(out)
        out = F.relu(out)
        out = self.bn(self.pool(out))
        out = self.linear(out)
        if self.emb_bn:
            out = self.bn2(out)
        return out4, out  # returns (frame_level, segment_level)


# ============================================================================
# BSRNN Legacy Model
# ============================================================================

class ResRNN(nn.Module):
    """Residual LSTM with GroupNorm + projection."""

    def __init__(self, input_size, hidden_size, bidirectional=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.eps = torch.finfo(torch.float32).eps
        self.norm = nn.GroupNorm(1, input_size, self.eps)
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True,
                           bidirectional=bidirectional)
        self.proj = nn.Linear(hidden_size * 2, input_size)

    def forward(self, input):
        rnn_output, _ = self.rnn(self.norm(input).transpose(1, 2).contiguous())
        rnn_output = self.proj(
            rnn_output.contiguous().view(-1, rnn_output.shape[2])
        ).view(input.shape[0], input.shape[2], input.shape[1])
        return input + rnn_output.transpose(1, 2).contiguous()


class BSNet(nn.Module):
    """Band-split network with intra-band and inter-band RNN."""

    def __init__(self, in_channel, nband=7, bidirectional=True):
        super().__init__()
        self.nband = nband
        self.feature_dim = in_channel // nband
        self.band_rnn = ResRNN(self.feature_dim, self.feature_dim * 2,
                               bidirectional=bidirectional)
        self.band_comm = ResRNN(self.feature_dim, self.feature_dim * 2,
                                bidirectional=bidirectional)

    def forward(self, input, dummy: Optional[torch.Tensor] = None):
        B, N, T = input.shape
        band_output = self.band_rnn(
            input.view(B * self.nband, self.feature_dim, -1)
        ).view(B, self.nband, -1, T)
        band_output = band_output.permute(0, 3, 2, 1).contiguous().view(
            B * T, -1, self.nband)
        output = self.band_comm(band_output).view(
            B, T, -1, self.nband).permute(0, 3, 2, 1).contiguous()
        return output.view(B, N, T)


class FuseSeparation(nn.Module):
    """Separation module with speaker fusion at each repeat."""

    def __init__(self, nband=7, num_repeat=6, feature_dim=128,
                 spk_emb_dim=256, spk_fuse_type="concat", multi_fuse=True):
        super().__init__()
        self.multi_fuse = multi_fuse
        self.nband = nband
        self.feature_dim = feature_dim
        self.separation = nn.ModuleList([])
        if self.multi_fuse:
            for _ in range(num_repeat):
                self.separation.append(
                    SpeakerFuseLayer(embed_dim=spk_emb_dim,
                                     feat_dim=feature_dim,
                                     fuse_type=spk_fuse_type))
                self.separation.append(BSNet(nband * feature_dim, nband))
        else:
            self.separation.append(
                SpeakerFuseLayer(embed_dim=spk_emb_dim,
                                 feat_dim=feature_dim,
                                 fuse_type=spk_fuse_type))
            for _ in range(num_repeat):
                self.separation.append(BSNet(nband * feature_dim, nband))

    def forward(self, x, spk_embedding, nch: torch.Tensor = torch.tensor(1)):
        batch_size = x.shape[0]
        if self.multi_fuse:
            for i, sep_func in enumerate(self.separation):
                x = sep_func(x, spk_embedding)
                if i % 2 == 0:
                    x = x.view(batch_size * nch,
                               self.nband * self.feature_dim, -1)
                else:
                    x = x.view(batch_size * nch, self.nband,
                               self.feature_dim, -1)
        else:
            x = self.separation[0](x, spk_embedding)
            x = x.view(batch_size * nch, self.nband * self.feature_dim, -1)
            for idx, sep in enumerate(self.separation):
                if idx > 0:
                    x = sep(x, spk_embedding)
            x = x.view(batch_size * nch, self.nband, self.feature_dim, -1)
        return x


class BSRNN(nn.Module):
    """Legacy BSRNN with joint speaker encoder (flat-config format).

    This is the exact model architecture used to train the PS4 checkpoint.
    State dict keys: separator.separation.*, spk_model.layer*, mask.*, BN.*,
    spk_encoder.*, preEmphasis.*
    """

    def __init__(
        self,
        spk_emb_dim=256,
        sr=16000,
        win=512,
        stride=128,
        feature_dim=128,
        num_repeat=6,
        use_spk_transform=True,
        use_bidirectional=True,
        spk_fuse_type="concat",
        multi_fuse=True,
        joint_training=True,
        multi_task=False,
        spksInTrain=251,
        spk_model=None,
        spk_model_init=None,
        spk_model_freeze=False,
        spk_args=None,
        spk_feat=False,
        feat_type="consistent",
    ):
        super().__init__()
        self.sr = sr
        self.win = win
        self.stride = stride
        self.group = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = feature_dim
        self.eps = torch.finfo(torch.float32).eps
        self.spk_emb_dim = spk_emb_dim
        self.joint_training = joint_training
        self.spk_feat = spk_feat
        self.feat_type = feat_type
        self.spk_model_freeze = spk_model_freeze
        self.multi_task = multi_task

        # Band split: 100Hz bins → 200Hz bins → 500Hz bins → 2kHz bins → rest
        bandwidth_100 = int(np.floor(100 / (sr / 2.0) * self.enc_dim))
        bandwidth_200 = int(np.floor(200 / (sr / 2.0) * self.enc_dim))
        bandwidth_500 = int(np.floor(500 / (sr / 2.0) * self.enc_dim))
        bandwidth_2k = int(np.floor(2000 / (sr / 2.0) * self.enc_dim))
        self.band_width = [bandwidth_100] * 15
        self.band_width += [bandwidth_200] * 10
        self.band_width += [bandwidth_500] * 5
        self.band_width += [bandwidth_2k] * 1
        self.band_width.append(self.enc_dim - int(np.sum(self.band_width)))
        self.nband = len(self.band_width)

        # Speaker embedding transform
        if use_spk_transform:
            self.spk_transform = SpeakerTransform()
        else:
            self.spk_transform = nn.Identity()

        # Joint speaker encoder
        if joint_training:
            spk_args = spk_args or {}
            self.spk_model = ECAPA_TDNN_GLOB_c512(
                feat_dim=spk_args.get("feat_dim", 80),
                embed_dim=spk_args.get("embed_dim", 192),
                pooling_func=spk_args.get("pooling_func", "ASTP"),
            )
            if spk_model_freeze:
                for param in self.spk_model.parameters():
                    param.requires_grad = False
            if not spk_feat:
                if feat_type == "consistent":
                    self.preEmphasis = PreEmphasis()
                    self.spk_encoder = torchaudio.transforms.MelSpectrogram(
                        sample_rate=sr,
                        n_fft=win,
                        win_length=win,
                        hop_length=stride,
                        f_min=20,
                        window_fn=torch.hamming_window,
                        n_mels=spk_args.get("feat_dim", 80),
                    )
            else:
                self.preEmphasis = nn.Identity()
                self.spk_encoder = nn.Identity()

            if multi_task:
                self.pred_linear = nn.Linear(spk_emb_dim, spksInTrain)
            else:
                self.pred_linear = nn.Identity()

        # Band normalization
        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            self.BN.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.band_width[i] * 2, self.eps),
                    nn.Conv1d(self.band_width[i] * 2, self.feature_dim, 1),
                )
            )

        # Separator
        self.separator = FuseSeparation(
            nband=self.nband,
            num_repeat=num_repeat,
            feature_dim=feature_dim,
            spk_emb_dim=spk_emb_dim,
            spk_fuse_type=spk_fuse_type,
            multi_fuse=multi_fuse,
        )

        # Mask estimation
        self.mask = nn.ModuleList([])
        for i in range(self.nband):
            self.mask.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.feature_dim, torch.finfo(torch.float32).eps),
                    nn.Conv1d(self.feature_dim, self.feature_dim * 4, 1),
                    nn.Tanh(),
                    nn.Conv1d(self.feature_dim * 4, self.feature_dim * 4, 1),
                    nn.Tanh(),
                    nn.Conv1d(self.feature_dim * 4, self.band_width[i] * 4, 1),
                )
            )

    def train(self, mode: bool = True):
        """Override train(): keep spk_model in eval mode when frozen."""
        super().train(mode)
        if self.spk_model_freeze and hasattr(self, "spk_model"):
            self.spk_model.eval()
        return self

    def forward(self, input, embeddings):
        """
        Args:
            input: (B, T) mixture waveform
            embeddings: (B, T_enroll) enrollment waveform (will be processed
                        by the internal speaker encoder, or (B, D) pre-extracted
                        speaker embedding if spk_feat=True)
        Returns:
            s: (B, T) extracted target speaker waveform
            _: dummy speaker label prediction (ignored at inference)
        """
        wav_input = input
        spk_emb_input = embeddings
        batch_size, nsample = wav_input.shape
        nch = 1

        # STFT
        spec = torch.stft(
            wav_input,
            n_fft=self.win,
            hop_length=self.stride,
            window=torch.hann_window(self.win).to(wav_input.device).type(
                wav_input.type()),
            return_complex=True,
        )
        spec_RI = torch.stack([spec.real, spec.imag], 1)

        # Band split
        subband_spec = []
        subband_mix_spec = []
        band_idx = 0
        for i in range(len(self.band_width)):
            subband_spec.append(
                spec_RI[:, :, band_idx:band_idx + self.band_width[i]].contiguous())
            subband_mix_spec.append(
                spec[:, band_idx:band_idx + self.band_width[i]])
            band_idx += self.band_width[i]

        # Band normalization
        subband_feature = []
        for i, bn_func in enumerate(self.BN):
            subband_feature.append(
                bn_func(subband_spec[i].view(batch_size * nch,
                                             self.band_width[i] * 2, -1)))
        subband_feature = torch.stack(subband_feature, 1)

        predict_speaker_lable = torch.tensor(0.0).to(spk_emb_input.device)

        # Joint speaker encoder
        if self.joint_training:
            if not self.spk_feat:
                if self.feat_type == "consistent":
                    with torch.no_grad():
                        spk_emb_input = self.preEmphasis(spk_emb_input)
                        spk_emb_input = self.spk_encoder(spk_emb_input) + 1e-8
                        spk_emb_input = spk_emb_input.log()
                        spk_emb_input = spk_emb_input - torch.mean(
                            spk_emb_input, dim=-1, keepdim=True)
                        spk_emb_input = spk_emb_input.permute(0, 2, 1)

            tmp_spk_emb_input = self.spk_model(spk_emb_input)
            if isinstance(tmp_spk_emb_input, tuple):
                spk_emb_input = tmp_spk_emb_input[-1]
            else:
                spk_emb_input = tmp_spk_emb_input
            predict_speaker_lable = self.pred_linear(spk_emb_input)

        spk_embedding = self.spk_transform(spk_emb_input)
        spk_embedding = spk_embedding.unsqueeze(1).unsqueeze(3)

        # Separation
        sep_output = self.separator(subband_feature, spk_embedding,
                                    torch.tensor(nch))

        # Mask estimation and complex mask application
        sep_subband_spec = []
        for i, mask_func in enumerate(self.mask):
            this_output = mask_func(sep_output[:, i]).view(
                batch_size * nch, 2, 2, self.band_width[i], -1)
            this_mask = this_output[:, 0] * torch.sigmoid(this_output[:, 1])
            this_mask_real = this_mask[:, 0]
            this_mask_imag = this_mask[:, 1]
            est_spec_real = (subband_mix_spec[i].real * this_mask_real
                             - subband_mix_spec[i].imag * this_mask_imag)
            est_spec_imag = (subband_mix_spec[i].real * this_mask_imag
                             + subband_mix_spec[i].imag * this_mask_real)
            sep_subband_spec.append(
                torch.complex(est_spec_real, est_spec_imag))

        # iSTFT
        est_spec = torch.cat(sep_subband_spec, 1)
        output = torch.istft(
            est_spec.view(batch_size * nch, self.enc_dim, -1),
            n_fft=self.win,
            hop_length=self.stride,
            window=torch.hann_window(self.win).to(wav_input.device).type(
                wav_input.type()),
            length=nsample,
        )
        output = output.view(batch_size, nch, -1)
        s = torch.squeeze(output, dim=1)
        return s, predict_speaker_lable


def ECAPA_TDNN_GLOB_c512(feat_dim, embed_dim, pooling_func="ASTP", emb_bn=False):
    """Factory function for ECAPA-TDNN with global context attention and 512 channels."""
    return ECAPA_TDNN(
        channels=512,
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        global_context_att=True,
        emb_bn=emb_bn,
    )


# ============================================================================
# Checkpoint Loading
# ============================================================================

def build_model(device: torch.device) -> BSRNN:
    """Build the PS4 BSRNN model with the exact training config parameters.

    Returns:
        BSRNN model in eval mode, moved to the specified device.
    """
    model = BSRNN(
        feat_type="consistent",
        feature_dim=128,
        num_repeat=6,
        spk_emb_dim=192,
        spk_fuse_type="multiply",
        multi_fuse=False,
        spk_model="ECAPA_TDNN_GLOB_c512",
        sr=16000,
        win=512,
        stride=128,
        spk_args={"feat_dim": 80, "embed_dim": 192, "pooling_func": "ASTP"},
        spk_model_freeze=True,
        use_spk_transform=False,
        joint_training=True,
        multi_task=False,
        spk_feat=False,
    )
    model.eval()
    return model.to(device)


def load_checkpoint(path: str, model: nn.Module, device: torch.device):
    """Load PS4 checkpoint weights into the model.

    The checkpoint is saved by train.py as:
        {"model": state_dict, "optimizer": ..., "scheduler": ..., ...}
    """
    print(f"[PS4] Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)

    # Handle various checkpoint formats
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[PS4]  WARNING: missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"[PS4]  WARNING: unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    print(f"[PS4]  Loaded successfully. "
          f"Epoch: {ckpt.get('epoch', 'N/A')}, "
          f"Step: {ckpt.get('step', 'N/A')}")
    return model


# ============================================================================
# Inference
# ============================================================================

def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load audio at target sample rate.

    Returns:
        Tensor of shape (1, T) — mono, normalized to [-1, 1].
    """
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)  # mono
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    # Normalize
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak
    return wav


def save_audio(path: str, wav: torch.Tensor, sr: int = 16000):
    """Save audio tensor to file."""
    torchaudio.save(path, wav.cpu(), sr)
    print(f"[PS4] Saved: {path}")


def extract_speaker(
    model: BSRNN,
    mixture: torch.Tensor,
    enrollment: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run target speaker extraction.

    Args:
        model: Loaded BSRNN model.
        mixture: (1, T_mix) mixture waveform, 16 kHz.
        enrollment: (1, T_enroll) enrollment waveform, 16 kHz.
        device: Computation device.

    Returns:
        (1, T_mix) extracted target speaker waveform.
    """
    with torch.no_grad():
        mixture = mixture.to(device)
        enrollment = enrollment.to(device)
        extracted, _ = model(mixture, enrollment)
    return extracted.cpu()


# ============================================================================
# CLI
# ============================================================================

def list_devices():
    """Print available CUDA devices."""
    print("Available devices:")
    print(f"  cpu")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  cuda:{i}  {torch.cuda.get_device_name(i)}")
    else:
        print("  (no CUDA devices found)")


def main():
    parser = argparse.ArgumentParser(
        description="PS4 Target Speaker Extraction — Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python inference.py --checkpoint checkpoint_epoch037.pt \\
      --mix mix.wav --enroll target.wav --output result.wav

  # Directory batch
  python inference.py --checkpoint checkpoint_epoch037.pt \\
      --mix-dir ./mixtures/ --enroll-dir ./enrollments/ --output-dir ./results/

  # List devices
  python inference.py --list-devices
        """,
    )
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoint_epoch037.pt",
                        help="Path to PS4 checkpoint (.pt)")
    parser.add_argument("--mix", type=str, default=None,
                        help="Path to mixture audio (16 kHz mono WAV)")
    parser.add_argument("--enroll", type=str, default=None,
                        help="Path to enrollment audio (16 kHz mono WAV)")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="Path to save extracted audio")
    parser.add_argument("--mix-dir", type=str, default=None,
                        help="Directory of mixture audio files (batch mode)")
    parser.add_argument("--enroll-dir", type=str, default=None,
                        help="Directory of enrollment audio files (batch mode, "
                             "must match mixture filenames)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (batch mode)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cpu', or 'cuda:N'")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available devices and exit")

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[PS4] Using device: {device}")

    # Build model
    print("[PS4] Building model...")
    model = build_model(device)
    load_checkpoint(args.checkpoint, model, device)
    print(f"[PS4] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Single file mode
    if args.mix is not None and args.enroll is not None:
        print(f"[PS4] Loading mixture: {args.mix}")
        mix = load_audio(args.mix)
        print(f"[PS4] Loading enrollment: {args.enroll}")
        enroll = load_audio(args.enroll)
        print(f"[PS4] Running extraction (mix: {mix.shape[-1]/16000:.1f}s, "
              f"enroll: {enroll.shape[-1]/16000:.1f}s)...")
        extracted = extract_speaker(model, mix, enroll, device)
        save_audio(args.output, extracted)
        return

    # Batch mode
    if args.mix_dir is not None and args.enroll_dir is not None and args.output_dir is not None:
        mix_dir = Path(args.mix_dir)
        enroll_dir = Path(args.enroll_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        mix_files = sorted(mix_dir.glob("*.wav"))
        if not mix_files:
            print(f"[PS4] No .wav files found in {mix_dir}")
            return

        print(f"[PS4] Batch mode: {len(mix_files)} files")
        for mix_path in mix_files:
            enroll_path = enroll_dir / mix_path.name
            if not enroll_path.exists():
                print(f"[PS4]  Skipping {mix_path.name}: no matching enrollment")
                continue
            out_path = output_dir / mix_path.name
            print(f"[PS4]  Processing {mix_path.name}...", end=" ", flush=True)
            mix = load_audio(str(mix_path))
            enroll = load_audio(str(enroll_path))
            extracted = extract_speaker(model, mix, enroll, device)
            save_audio(str(out_path), extracted)
            print("done")
        return

    # If neither mode is specified
    parser.print_help()
    print("\n[PS4] ERROR: Specify either --mix/--enroll (single) or "
          "--mix-dir/--enroll-dir/--output-dir (batch).")


if __name__ == "__main__":
    main()