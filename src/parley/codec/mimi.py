"""
Thin, research-friendly wrapper around kyutai/mimi via transformers.

Typical usage:
    codec = MimiCodec()
    codes = codec.encode(wave_16k)          # (num_codebooks, num_frames) int64
    recon = codec.decode(codes)             # (T,) float32 16kHz

Mimi frame rate = 12.5 Hz (80 ms frames).
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio
from transformers import AutoFeatureExtractor, MimiModel

from parley.rows.schema import FRAME_MS, SAMPLE_RATE

MIMI_REPO = "kyutai/mimi"
#: The revision speak-clean-heart-v1.0 was coded with.
MIMI_REVISION = "89091b3e466eb6a9d11e537bf26b144f194978f7"
MIMI_SR = 24_000
MIMI_SAMPLES_PER_FRAME = MIMI_SR * FRAME_MS // 1000


def normalize_audio(audio: np.ndarray | torch.Tensor) -> np.ndarray:
    """Float32 mono, scaled down into [-1, 1] only when it peaks above 1."""
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    peak = np.abs(audio).max()
    if peak > 1.0:
        audio = audio / peak
    return audio


class MimiCodec:
    #: How many of Mimi's RVQ levels the LLM actually models. Mimi exposes 32; the
    #: path-of-record uses the first 8 (Moshi-style). Measured 2026-07-24: on our own
    #: Kokoro output distribution 8 is effectively transparent (codec-induced drift 0.014
    #: and 32 is no better), and a blind listen agreed — 16 vs 32 indistinguishable, 8 vs 16
    #: barely perceptible. On real human speech 8 costs 0.27 drift versus 0.07 at 32, so
    #: this becomes a live lever the moment training audio stops being synthetic. Sesame CSM
    #: and MisoTTS both use 32. Raising it costs one extra sequential depth-decode step per
    #: codebook, so pair any increase with the KV-cached decode.
    DEFAULT_EFFECTIVE_CODEBOOKS = 8

    def __init__(
        self,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        effective_codebooks: int | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32

        print(f"[MimiCodec] Loading {MIMI_REPO} on {self.device} ...")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(MIMI_REPO, revision=MIMI_REVISION)
        self.model = MimiModel.from_pretrained(MIMI_REPO, revision=MIMI_REVISION).to(self.device).eval()
        if dtype is not None:
            self.model = self.model.to(dtype=self.dtype)

        # Inspect config
        cfg = self.model.config
        raw_cb = getattr(cfg, "num_codebooks", getattr(cfg, "num_quantizers", 32))
        self.num_codebooks = int(getattr(cfg, "effective_codebooks", raw_cb))
        requested = self.DEFAULT_EFFECTIVE_CODEBOOKS if effective_codebooks is None else int(effective_codebooks)
        if requested < 1 or requested > self.num_codebooks:
            raise ValueError(f"effective_codebooks must be in [1, {self.num_codebooks}]; got {requested}")
        self.effective_codebooks = requested
        self.frame_rate = getattr(cfg, "sampling_rate", 24000)
        self.codec_sr = getattr(self.feature_extractor, "sampling_rate", 24000)

        print(f"[MimiCodec] raw_codebooks={raw_cb} effective={self.effective_codebooks} codec_sr={self.codec_sr}")

    @torch.inference_mode()
    def encode(self, audio_16k: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Encode 16 kHz waveform to Mimi codes.

        Returns:
            codes: (num_codebooks, num_frames) long tensor on CPU
        """
        wav = self._prepare_input(audio_16k)  # (1, T) at codec_sr
        inputs = self.feature_extractor(
            raw_audio=wav.squeeze(0).cpu().numpy(),
            sampling_rate=self.codec_sr,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self.device, dtype=self.dtype)

        encoder_outputs = self.model.encode(input_values)
        codes = encoder_outputs.audio_codes[0].detach().cpu()  # (C, F)
        if codes.shape[0] > self.effective_codebooks:
            codes = codes[: self.effective_codebooks]
        return codes.long()

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> np.ndarray:
        """
        Decode Mimi codes back to waveform.

        codes: (num_codebooks, num_frames) or (1, C, F)
        Returns 16 kHz (resampled if codec_sr != 16k) float32 mono.
        """
        if codes.ndim == 2:
            codes = codes.unsqueeze(0)  # (1, C, F)
        codes = codes.to(self.device)

        audio_values = self.model.decode(codes)["audio_values"]  # (1, 1, T_codec)
        audio = audio_values[0, 0].detach().cpu().float().numpy()

        # Resample back to our target 16 kHz if necessary
        if self.codec_sr != SAMPLE_RATE:
            wav = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
            resampler = torchaudio.transforms.Resample(orig_freq=self.codec_sr, new_freq=SAMPLE_RATE)
            audio = resampler(wav).squeeze(0).numpy()
        return normalize_audio(audio)

    def _prepare_input(self, audio: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Ensure we feed the codec at its expected sampling rate."""
        audio = normalize_audio(audio)
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        audio = audio.float()

        if self.codec_sr != SAMPLE_RATE:
            wav = audio.unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=SAMPLE_RATE, new_freq=self.codec_sr)
            wav = resampler(wav)
            audio = wav.squeeze(0)
        return audio.unsqueeze(0)  # (1, T)


@torch.inference_mode()
def mimi_codes(mimi: MimiModel, wav24: torch.Tensor, *, frames: int, codebooks: int = 8) -> np.ndarray:
    """[frames*1920] 24 kHz float -> uint16 [frames, codebooks]."""
    if wav24.shape[0] != frames * MIMI_SAMPLES_PER_FRAME:
        raise ValueError(f"expected {frames * MIMI_SAMPLES_PER_FRAME} samples, got {wav24.shape[0]}")
    out = mimi.encode(wav24[None, None, :].float())
    codes = out.audio_codes[0, :codebooks]
    if codes.shape[1] < frames:
        raise RuntimeError(f"Mimi returned {codes.shape[1]} frames for {frames}")
    return codes[:, :frames].T.contiguous().cpu().numpy().astype(np.uint16)
