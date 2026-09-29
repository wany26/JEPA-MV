#!/usr/bin/env python3
"""Synthesize the soundtrack for the JEPA music video.

Everything is generated from code: drums, bass, pads, arps and FX are
synthesized with numpy, the narration is spoken by a Piper TTS voice, and the
"robot" hook is a Piper voice pushed through a channel vocoder whose carrier
plays the melody, so the robot actually sings.

Reads  src/song.json
Writes build/soundtrack.wav, build/stems/*.wav (for QA) and mv/timing.js
       (lyric/word timings, beat grid and an audio-energy envelope that the
       visuals use to stay in sync with the music).
"""

import json
import os
import re
import sys
import urllib.request
import wave

import numpy as np
from scipy import signal as sps
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")
VOICE_DIR = os.path.join(BUILD, "voices")
SR = 44100
VOICES = {
    "narrator": "en_US-lessac-high",
    "robot": "en_US-ryan-high",
}
VOICE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/{family}/{lang}/{spk}/{q}/{name}{ext}"

song = json.load(open(os.path.join(ROOT, "src", "song.json")))
BPM = song["bpm"]
BEAT = 60.0 / BPM
BAR = 4 * BEAT
STEP = BEAT / 4
DUR = float(song["duration"])
N = int(DUR * SR)
TAIL = 4 * SR  # room for reverb tails before the final fade

rng_master = np.random.default_rng(20220627)  # date of LeCun's position paper


# --------------------------------------------------------------------------
# Song structure helpers
# --------------------------------------------------------------------------
PC = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7,
      "G#": 8, "A": 9, "A#": 10, "B": 11}

BAR_CHORDS = []
BAR_SECTION = []
for sec in song["sections"]:
    assert len(sec["chords"]) == sec["bars"], sec["name"]
    BAR_CHORDS += sec["chords"]
    BAR_SECTION += [sec["name"]] * sec["bars"]
NBARS = len(BAR_CHORDS)
SEC_START = {s["name"]: s["bar"] for s in song["sections"]}


def parse_chord(name):
    m = re.match(r"([A-G]#?)(m?)$", name)
    root = PC[m.group(1)]
    third = 3 if m.group(2) == "m" else 4
    return root, [0, third, 7]


def voicing(root, ivs, lo):
    """Each chord tone placed in the octave [lo, lo+12)."""
    return sorted(((root + iv - lo) % 12) + lo for iv in ivs)


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def bar_t(bar, beat=0.0):
    return bar * BAR + beat * BEAT


# --------------------------------------------------------------------------
# DSP primitives
# --------------------------------------------------------------------------
def stereo(n=N + TAIL):
    return np.zeros((2, n), dtype=np.float64)


def place(buf, x, t, gain=1.0, pan=0.0):
    """Add mono (or stereo) signal x into stereo buffer at time t."""
    i = int(round(t * SR))
    if i < 0:
        x = x[..., -i:]
        i = 0
    n = min(x.shape[-1], buf.shape[1] - i)
    if n <= 0:
        return
    if x.ndim == 1:
        a = (pan + 1) * np.pi / 4
        buf[0, i:i + n] += x[:n] * gain * np.cos(a) * np.sqrt(2)
        buf[1, i:i + n] += x[:n] * gain * np.sin(a) * np.sqrt(2)
    else:
        buf[:, i:i + n] += x[:, :n] * gain


def env_adsr(n, a=0.005, d=0.1, s=0.7, r=0.05, gate=None):
    t = np.arange(n) / SR
    gate = n / SR if gate is None else gate
    e = np.where(t < a, t / max(a, 1e-6),
                 s + (1 - s) * np.exp(-(t - a) / max(d, 1e-6)))
    rel = np.clip(1 - (t - gate) / max(r, 1e-6), 0, 1)
    return e * np.where(t > gate, rel, 1.0)


def fade(x, fin=0.003, fout=0.01):
    n = x.shape[-1]
    a, b = int(fin * SR), int(fout * SR)
    if a:
        x[..., :a] *= np.linspace(0, 1, a)
    if b:
        x[..., n - b:] *= np.linspace(1, 0, b)
    return x


def butter(x, fc, btype="low", order=2):
    sos = sps.butter(order, fc, btype, fs=SR, output="sos")
    return sps.sosfilt(sos, x, axis=-1)


def peaking(x, f0, gain_db, q=1.0):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / SR
    al = np.sin(w0) / (2 * q)
    b = [1 + al * A, -2 * np.cos(w0), 1 - al * A]
    a = [1 + al / A, -2 * np.cos(w0), 1 - al / A]
    return sps.lfilter(b, a, x, axis=-1)


def tv_filter(x, fc_fn, btype="low", block=512, order=2):
    """Time-varying Butterworth filter (block-wise coefficient updates)."""
    x = np.atleast_2d(x)
    out = np.zeros_like(x)
    zi = None
    for i in range(0, x.shape[1], block):
        fc = float(np.clip(fc_fn((i + block / 2) / SR), 20, SR / 2 - 200))
        if btype == "band":
            fc = [fc / 1.6, min(fc * 1.6, SR / 2 - 100)]
        sos = sps.butter(order, fc, btype, fs=SR, output="sos")
        if zi is None:
            zi = np.zeros((x.shape[0], sos.shape[0], 2))
        for c in range(x.shape[0]):
            out[c, i:i + block], zi[c] = sps.sosfilt(sos, x[c, i:i + block], zi=zi[c])
    return out


def saw_blep(freq, n, phase0=0.0):
    """Band-limited (polyBLEP) sawtooth. freq may be scalar or per-sample."""
    dt = np.broadcast_to(np.asarray(freq, dtype=np.float64) / SR, (n,))
    ph = (phase0 + np.cumsum(dt)) % 1.0
    y = 2 * ph - 1
    m = ph < dt
    t = ph[m] / dt[m]
    y[m] -= t + t - t * t - 1
    m = ph > 1 - dt
    t = (ph[m] - 1) / dt[m]
    y[m] -= t * t + t + t + 1
    return y


def additive(f0, n, amp_fn, kmax=60, phases=None):
    """Sum of harmonics k*f0 with time-varying amplitude amp_fn(k, t)."""
    kmax = int(max(1, min(kmax, (SR / 2 - 500) // f0)))
    t = np.arange(n) / SR
    k = np.arange(1, kmax + 1)[:, None]
    ph = 0 if phases is None else phases[:kmax, None]
    return (amp_fn(k, t[None, :]) * np.sin(2 * np.pi * k * f0 * t[None, :] + ph)).sum(0)


def make_ir(rt60=2.6, seed=1, predelay=0.025, damp=0.5):
    rng = np.random.default_rng(seed)
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    ir = np.zeros((2, n + int(predelay * SR)))
    for c in range(2):
        noise = rng.standard_normal(n)
        bright = butter(noise, 7000)
        dark = butter(noise, 1800)
        mix = (1 - damp * t / rt60) * bright + damp * (t / rt60) * dark
        ir[c, int(predelay * SR):] = mix * np.exp(-6.91 * t / rt60)
    return ir / np.sqrt((ir ** 2).sum() / 2)


def reverb(x, ir):
    return np.stack([sps.oaconvolve(x[c], ir[c])[: x.shape[1]] for c in range(2)])


def pingpong(x, delay, fb=0.45, taps=6, lp=4500):
    mono = x.mean(0)
    out = np.zeros_like(x)
    d = int(delay * SR)
    for i in range(1, taps + 1):
        g = fb ** (i - 1)
        c = (i - 1) % 2
        out[c, d * i:] += mono[: out.shape[1] - d * i] * g
    return butter(out, lp)


def smooth_env(x, cutoff=10.0):
    return np.maximum(butter(np.abs(x), cutoff, order=1), 0)


def rms_db(x):
    return 10 * np.log10(np.mean(x ** 2) + 1e-12)


def active_rms_db(x):
    """RMS over the parts of a stem where it is actually playing."""
    m = x.mean(0) if x.ndim == 2 else x
    e = uniform_filter1d(m ** 2, SR // 10)
    thr = e.max() * 1e-3
    act = e > thr
    if not act.any():
        return -120.0
    return 10 * np.log10(e[act].mean() + 1e-12)


# --------------------------------------------------------------------------
# Drum sounds
# --------------------------------------------------------------------------
def snd_kick():
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    f = 43 + 120 * np.exp(-t / 0.032) + 250 * np.exp(-t / 0.004)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.30)
    click = butter(rng_master.standard_normal(n), 1800, "high") * np.exp(-t / 0.0025) * 0.35
    x = np.tanh(1.8 * (body + click)) / np.tanh(1.8)
    return fade(x, 0, 0.03)


def snd_snare():
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    tone = (np.sin(2 * np.pi * 186 * t) * np.exp(-t / 0.06) * 0.55
            + np.sin(2 * np.pi * 332 * t) * np.exp(-t / 0.04) * 0.25)
    noise = sps.sosfilt(sps.butter(2, [1400, 9500], "band", fs=SR, output="sos"),
                        rng_master.standard_normal(n)) * np.exp(-t / 0.15)
    return fade(np.tanh(1.5 * (tone + noise * 0.9)), 0, 0.02)


def snd_clap():
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    e = sum(np.where(t >= o, np.exp(-(t - o) / 0.005), 0) for o in (0, 0.011, 0.023))
    e = e + np.where(t >= 0.03, np.exp(-(t - 0.03) / 0.13), 0) * 0.8
    noise = sps.sosfilt(sps.butter(2, [900, 5200], "band", fs=SR, output="sos"),
                        rng_master.standard_normal(n))
    return fade(noise * e, 0, 0.02)


def snd_hat(decay):
    n = int((decay * 6 + 0.02) * SR)
    t = np.arange(n) / SR
    metal = sum(np.sign(np.sin(2 * np.pi * f * 1.6 * t)) for f in (205.3, 304.4, 369.6, 522.7, 540, 800))
    x = 0.55 * butter(rng_master.standard_normal(n), 7000, "high") + 0.25 * butter(metal, 7500, "high")
    return fade(x * np.exp(-t / decay), 0.0005, 0.01)


def snd_crash():
    n = int(3.0 * SR)
    t = np.arange(n) / SR
    metal = sum(np.sign(np.sin(2 * np.pi * f * 2.3 * t + i)) for i, f in enumerate((205.3, 304.4, 369.6, 522.7, 540, 800)))
    x = butter(rng_master.standard_normal(n), 3500, "high") * 0.7 + butter(metal, 5000, "high") * 0.2
    return fade(x * np.exp(-t / 0.9), 0.001, 0.2)


KICK, SNARE, CLAP = snd_kick(), snd_snare(), snd_clap()
HAT_C, HAT_O, CRASH = snd_hat(0.03), snd_hat(0.22), snd_crash()


# --------------------------------------------------------------------------
# Tonal instruments
# --------------------------------------------------------------------------
_cache = {}


def bass_note(m, dur, bright=1.0):
    key = ("bass", m, round(dur, 3), round(bright, 2))
    if key in _cache:
        return _cache[key]
    f0 = mtof(m)
    n = int((dur + 0.03) * SR)

    def amp(k, t):
        cut = 140 + bright * 1500 * np.exp(-t / 0.09)
        return (1.0 / k) * 1 / np.sqrt(1 + ((k * f0) / cut) ** 4)

    x = additive(f0, n, amp, kmax=48)
    x += 0.6 * np.sin(2 * np.pi * f0 * np.arange(n) / SR)  # sub
    x *= env_adsr(n, a=0.003, d=0.12, s=0.75, r=0.03, gate=dur)
    x = np.tanh(1.3 * x)
    _cache[key] = fade(x, 0.002, 0.005)
    return _cache[key]


def pluck(m, dur=0.4, bright=1.0):
    key = ("pluck", m, round(dur, 3), round(bright, 2))
    if key in _cache:
        return _cache[key]
    f0 = mtof(m)
    n = int(dur * SR)

    def amp(k, t):
        odd = np.where(k % 2 == 1, 1.0, 0.35)
        return odd / k * np.exp(-t * (4 + 2.6 * k / bright))

    x = additive(f0, n, amp, kmax=30)
    _cache[key] = fade(x, 0.001, 0.02)
    return _cache[key]


def pad_chord(notes, dur, seed):
    """Detuned supersaw chord, stereo."""
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    out = np.zeros((2, n))
    t = np.arange(n) / SR
    for j, m in enumerate(notes):
        f0 = mtof(m)
        for v in range(5):
            det = (v - 2) * 0.09  # semitones
            lfo = 1 + 0.0015 * np.sin(2 * np.pi * (0.3 + 0.07 * v) * t + rng.uniform(0, 6.28))
            x = saw_blep(mtof(m + det) * lfo, n, rng.uniform())
            pan = [-0.8, -0.4, 0.0, 0.4, 0.8][v]
            a = (pan + 1) * np.pi / 4
            out[0] += x * np.cos(a)
            out[1] += x * np.sin(a)
    return out / (5 * len(notes)) * 2.5


def lead_note(m, dur, prev=None):
    n = int((dur + 0.15) * SR)
    t = np.arange(n) / SR
    f = mtof(m) * (1 + 0.006 * np.sin(2 * np.pi * 5.5 * t) * np.clip((t - 0.2) / 0.3, 0, 1))
    if prev is not None:
        f = f * 2 ** ((prev - m) / 12 * np.exp(-t / 0.04))
    x = saw_blep(f, n) + 0.5 * saw_blep(f * 1.005, n, 0.3)
    x = butter(x, 3200) * env_adsr(n, 0.01, 0.3, 0.8, 0.12, gate=dur)
    return fade(x, 0.003, 0.02)


# --------------------------------------------------------------------------
# FX
# --------------------------------------------------------------------------
def riser(dur, f_lo=300, f_hi=9000, seed=5):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    noise = rng.standard_normal((2, n))
    x = tv_filter(noise, lambda tt: f_lo * (f_hi / f_lo) ** (min(tt, dur) / dur) ** 2, "band", order=2)
    sweep = saw_blep(110 * 2 ** (3 * (t / dur) ** 1.5), n) * 0.25
    x = x + butter(sweep, 5000)[None, :]
    return x * ((t / dur) ** 2.2)[None, :]


def impact(seed=7):
    rng = np.random.default_rng(seed)
    n = int(2.5 * SR)
    t = np.arange(n) / SR
    boom = np.sin(2 * np.pi * np.cumsum(30 + 60 * np.exp(-t / 0.18)) / SR) * np.exp(-t / 0.8)
    hit = butter(rng.standard_normal(n), 2500) * np.exp(-t / 0.12) * 0.6
    return fade(np.tanh(1.4 * (boom + hit)), 0, 0.3)


def downlifter(dur=2.5, seed=9):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = tv_filter(rng.standard_normal((2, n)), lambda tt: 8000 * (200 / 8000) ** min(tt / dur, 1), "band")
    return x * np.exp(-t / (dur / 3))[None, :]


def reverse_cymbal(dur=2.0):
    c = CRASH[: int(dur * SR)][::-1].copy()
    return fade(c, 0.2, 0.005)


# --------------------------------------------------------------------------
# Voices: Piper TTS + channel vocoder
# --------------------------------------------------------------------------
def ensure_voice(name):
    os.makedirs(VOICE_DIR, exist_ok=True)
    lang, spk, q = name.split("-")
    for ext in (".onnx", ".onnx.json"):
        path = os.path.join(VOICE_DIR, name + ext)
        if not os.path.exists(path):
            url = VOICE_URL.format(family=lang.split("_")[0], lang=lang, spk=spk, q=q, name=name, ext=ext)
            print("downloading", url, file=sys.stderr)
            urllib.request.urlretrieve(url, path)
    return os.path.join(VOICE_DIR, name + ".onnx")


_voices = {}


def get_voice(role):
    from piper import PiperVoice
    if role not in _voices:
        _voices[role] = PiperVoice.load(ensure_voice(VOICES[role]), include_alignments=True)
    return _voices[role]


PUNCT = set(",.?!;:")


def tts(role, text, length_scale=1.0, gap=0.16):
    """Return (audio@44.1k, [(word_start, word_end), ...]) with silence trimmed."""
    from piper import SynthesisConfig
    voice = get_voice(role)
    cfg = SynthesisConfig(length_scale=length_scale, noise_scale=0.55, noise_w_scale=0.7)
    pieces, words, pos = [], [], 0
    for chunk in voice.synthesize(text, syn_config=cfg, include_alignments=True):
        a = chunk.audio_float_array.astype(np.float64)
        sr = chunk.sample_rate
        cur, wstart = 0, None
        for al in chunk.phoneme_alignments or []:
            p, ns = al.phoneme, int(al.num_samples)
            if p in ("^", "$", "_") or p == " ":
                if p == " " and wstart is not None:
                    words.append(((pos + wstart) / sr, (pos + cur) / sr))
                    wstart = None
            elif p not in PUNCT and p not in ("ˈ", "ˌ") and wstart is None:
                wstart = cur
            if p in PUNCT and wstart is not None:
                words.append(((pos + wstart) / sr, (pos + cur) / sr))
                wstart = None
            cur += ns
        if wstart is not None:
            words.append(((pos + wstart) / sr, (pos + cur) / sr))
        pieces.append(a)
        pieces.append(np.zeros(int(gap * sr)))
        pos += len(a) + int(gap * sr)
    audio = np.concatenate(pieces[:-1])
    sr = 22050
    # trim silence
    env = uniform_filter1d(np.abs(audio), 64)
    idx = np.where(env > 0.012)[0]
    s0, s1 = max(idx[0] - 32, 0), min(idx[-1] + 400, len(audio))
    audio = audio[s0:s1]
    off = s0 / sr
    words = [(max(a - off, 0), max(b - off, 0)) for a, b in words]
    audio = sps.resample_poly(audio, 2, 1)
    return audio, words


# Vocoder ---------------------------------------------------------------
VN, VH = 1024, 256
VWIN = np.hanning(VN + 1)[:-1]


def stft(x):
    x = np.concatenate([np.zeros(VN // 2), x, np.zeros(VN)])
    nfr = 1 + (len(x) - VN) // VH
    idx = np.arange(VN)[None, :] + VH * np.arange(nfr)[:, None]
    return np.fft.rfft(x[idx] * VWIN, axis=1).T  # bins x frames


def istft(X, n):
    frames = np.fft.irfft(X.T, n=VN, axis=1) * VWIN
    out = np.zeros(VH * (frames.shape[0] - 1) + VN)
    wsum = np.zeros_like(out)
    for i, fr in enumerate(frames):
        out[i * VH:i * VH + VN] += fr
        wsum[i * VH:i * VH + VN] += VWIN ** 2
    out /= np.maximum(wsum, 1e-3)
    return out[VN // 2: VN // 2 + n]


def band_matrix(nb=40, lo=70, hi=15000):
    freqs = np.fft.rfftfreq(VN, 1 / SR)
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = imel(np.linspace(mel(lo), mel(hi), nb + 2))
    B = np.zeros((nb, len(freqs)))
    for b in range(nb):
        l, c, h = edges[b], edges[b + 1], edges[b + 2]
        B[b] = np.clip(np.minimum((freqs - l) / (c - l), (h - freqs) / (h - c)), 0, None)
    Binterp = B / np.maximum(B.sum(0, keepdims=True), 1e-9)  # bins interpolate band gains
    return B, Binterp


VB, VBI = band_matrix()


def vocode(mod, target_len, notes_fn, seed=0, detune=0.0):
    """Channel vocoder with voicing-aware time stretch of the modulator.

    mod        : modulator speech at SR
    target_len : output length in seconds (speech is stretched to fit)
    notes_fn   : function(t_array) -> list of (freq_array, gain) carrier partials
    """
    rng = np.random.default_rng(seed)
    M = stft(mod)
    P = np.abs(M) ** 2
    E = VB @ P  # bands x frames
    Tm = E.shape[1]
    # voicing weight: loud frames with energy mostly below 1.5 kHz are vowels
    low = E[:14].sum(0)
    tot = E.sum(0) + 1e-12
    loud = np.sqrt(tot / tot.max())
    voiced = np.clip((low / tot - 0.35) / 0.5, 0, 1) * np.clip(loud * 2.5, 0, 1)
    n_out = int(target_len * SR)
    To = 1 + n_out // VH
    ratio = To / Tm
    if ratio > 1:
        # stretch vowels more than consonants
        a = (ratio * Tm - Tm) / max(voiced.sum(), 1e-6)
        w = 1 + a * voiced
    else:
        w = np.full(Tm, ratio)
    cum = np.concatenate([[0], np.cumsum(w)])
    cum *= To / cum[-1]
    src = np.interp(np.arange(To), cum[:-1], np.arange(Tm))
    logE = np.log(E + 1e-10)
    i0 = np.floor(src).astype(int)
    fr = src - i0
    i1 = np.minimum(i0 + 1, Tm - 1)
    Eo = np.exp(logE[:, i0] * (1 - fr) + logE[:, i1] * fr)
    Eo = uniform_filter1d(Eo, 3, axis=1)

    # carrier
    t = np.arange(n_out) / SR
    car = np.zeros(n_out)
    for freq, g in notes_fn(t):
        car += g * saw_blep(freq * 2 ** (detune / 1200), n_out, rng.uniform())
    car += 0.22 * butter(rng.standard_normal(n_out), 2500, "high")
    C = stft(car)[:, :To]
    Ec = VB @ (np.abs(C) ** 2)
    gain = np.sqrt(Eo[:, : C.shape[1]] / (Ec + 1e-10))
    Y = C * (VBI.T @ gain)
    y = istft(Y, n_out)
    y = y / (np.abs(y).max() + 1e-9)
    return fade(y, 0.004, 0.03)


# --------------------------------------------------------------------------
# Arrangement
# --------------------------------------------------------------------------
def pattern_hits(pat, bar):
    return [bar_t(bar) + i * STEP for i, ch in enumerate(pat) if ch != "."]


def build_drums():
    kick, snare, hats, crash = stereo(), stereo(), stereo(), stereo()
    kick_times = []
    rng = np.random.default_rng(3)

    def K(t, g=1.0):
        place(kick, KICK, t, g)
        kick_times.append(t)

    for bar in range(NBARS):
        sec = BAR_SECTION[bar]
        rel = bar - SEC_START[sec]
        if sec in ("verse1", "verse2"):
            kp = "x.....x...x....." if rel % 4 != 3 else "x.....x...x...x."
            for t in pattern_hits(kp, bar):
                K(t, 0.9)
            for t in pattern_hits("....x.......x...", bar):
                place(snare, SNARE, t, 0.8)
                place(snare, CLAP, t, 0.35, 0.1)
            hp = "x.x.x.x.x.x.x.x." if sec == "verse1" else "xxx.xxx.xxx.xxxx"
            for i, t in enumerate(pattern_hits(hp, bar)):
                place(hats, HAT_C, t, (0.55 if i % 2 == 0 else 0.35) * rng.uniform(0.85, 1.0), 0.3)
            if rel == 15:
                for t in pattern_hits("............xxxx", bar):
                    place(snare, SNARE, t, 0.45, -0.2)
        elif sec == "pre":
            if rel < 7:
                for t in pattern_hits("x...x...x...x...", bar):
                    K(t, 0.95)
                for t in pattern_hits("....x.......x...", bar):
                    place(snare, SNARE, t, 0.8)
                for t in pattern_hits("..x...x...x...x.", bar):
                    place(hats, HAT_O, t, 0.35, 0.25)
                if rel in (4, 5):
                    for i, t in enumerate(pattern_hits("x.x.x.x.x.x.x.x.", bar)):
                        place(snare, SNARE, t, 0.25 + 0.1 * i / 8, -0.15)
                if rel == 6:
                    for i, t in enumerate(pattern_hits("xxxxxxxxxxxxxxxx", bar)):
                        place(snare, SNARE, t, 0.3 + 0.5 * i / 16, 0.15 * (-1) ** i)
        elif sec in ("chorus1", "chorus2"):
            if rel < 14:
                for t in pattern_hits("x...x...x...x...", bar):
                    K(t, 1.0)
                for t in pattern_hits("....x.......x...", bar):
                    place(snare, SNARE, t, 0.85)
                    place(snare, CLAP, t, 0.6, -0.1)
                for t in pattern_hits("..x...x...x...x.", bar):
                    place(hats, HAT_O, t, 0.4, 0.3)
                for i, t in enumerate(pattern_hits("xxxxxxxxxxxxxxxx", bar)):
                    place(hats, HAT_C, t, (0.28 if i % 2 else 0.18) * rng.uniform(0.8, 1), -0.35)
                if sec == "chorus2":
                    for t in pattern_hits(".......x.......x", bar):
                        place(snare, CLAP, t, 0.25, 0.35)
                if rel in (0, 8):
                    place(crash, CRASH, bar_t(bar), 0.8)
            else:
                # "J - E - P - A": hits on each letter (half notes)
                for b in (0, 2):
                    K(bar_t(bar, b), 1.0)
                    place(snare, SNARE, bar_t(bar, b), 0.6)
                    place(crash, CRASH, bar_t(bar, b), 0.45, 0.3 * (-1) ** b)
        elif sec == "bridge":
            if 4 <= rel < 7:
                for t in pattern_hits("x...x...x...x...", bar):
                    K(t, 0.9)
                for t in pattern_hits("..x...x...x...x.", bar):
                    place(hats, HAT_C, t, 0.35, 0.3)
                if rel == 6:
                    for i, t in enumerate(pattern_hits("xxxxxxxxxxxxxxxx", bar)):
                        place(snare, SNARE, t, 0.25 + 0.55 * i / 16, 0.15 * (-1) ** i)
                elif rel == 5:
                    for i, t in enumerate(pattern_hits("x.x.x.x.x.x.x.x.", bar)):
                        place(snare, SNARE, t, 0.25 + 0.15 * i / 8, -0.15)
        elif sec == "outro":
            if rel < 4:
                for t in pattern_hits("x.......x.......", bar):
                    K(t, 0.8 - 0.1 * rel)
                for t in pattern_hits("..x...x...x...x.", bar):
                    place(hats, HAT_C, t, 0.3 - 0.05 * rel, 0.3)
                if rel == 0:
                    place(crash, CRASH, bar_t(bar), 0.6)
    # letter hits on the short "J E P A" breaks (quarter notes)
    for L in song["robot"]["letters"]:
        if L["t"] < 16:
            continue
        for i in range(4):
            K(L["t"] + i * BEAT, 0.8 + 0.05 * i)
    return dict(kick=kick, snare=snare, hats=hats, crash=crash), sorted(kick_times)


def build_bass():
    bass = stereo()
    for bar in range(NBARS):
        sec = BAR_SECTION[bar]
        rel = bar - SEC_START[sec]
        root, _ = parse_chord(BAR_CHORDS[bar])
        m = 24 + root
        if m < 28:
            m += 12
        if sec == "intro":
            if rel >= 4:
                place(bass, bass_note(m, BAR * 0.95, 0.3), bar_t(bar), 0.8)
        elif sec in ("verse1", "verse2"):
            for i in range(8):
                acc = 1.0 if i % 2 == 0 else 0.75
                place(bass, bass_note(m, STEP * 1.7, 0.7 * acc), bar_t(bar, i * 0.5), acc)
        elif sec == "pre":
            if rel < 4:
                for i in range(8):
                    place(bass, bass_note(m, STEP * 1.7, 0.8), bar_t(bar, i * 0.5), 1.0 if i % 2 == 0 else 0.8)
            elif rel < 7:
                for i in range(16):
                    place(bass, bass_note(m + (12 if i % 4 == 2 else 0), STEP * 0.85, 0.9), bar_t(bar, i * 0.25), 0.9)
        elif sec in ("chorus1", "chorus2"):
            if rel < 14:
                for i in range(16):
                    oct_ = 12 if i % 4 == 2 else 0
                    place(bass, bass_note(m + oct_, STEP * 0.85, 1.0), bar_t(bar, i * 0.25), 1.0 if i % 4 == 0 else 0.85)
            else:
                for b in (0, 2):
                    place(bass, bass_note(m, BEAT * 1.8, 1.0), bar_t(bar, b), 1.0)
        elif sec == "bridge":
            if rel < 4:
                place(bass, bass_note(m, BAR * 0.95, 0.35), bar_t(bar), 0.8)
            elif rel < 7:
                for i in range(8):
                    place(bass, bass_note(m, STEP * 1.7, 0.8), bar_t(bar, i * 0.5), 0.95)
        elif sec == "outro":
            if rel < 4:
                for i in range(8):
                    place(bass, bass_note(m, STEP * 1.7, 0.6), bar_t(bar, i * 0.5), 0.9 - 0.1 * rel)
            elif rel == 4:
                place(bass, bass_note(m, BAR * 3.5, 0.25), bar_t(bar), 0.7)
    for L in song["robot"]["letters"]:
        if L["t"] < 16:
            continue
        m = 24 + (L["notes"][0] % 12)
        m = m + 12 if m < 28 else m
        for i in range(4):
            place(bass, bass_note(m, BEAT * 0.9, 1.0), L["t"] + i * BEAT, 0.9)
    return bass


def pad_cutoff(t):
    b = t / BAR
    if b < 8:
        return 700 + 1800 * (b / 8) ** 1.5
    if b < 24:
        return 2000
    if b < 32:
        return 1400 + 5000 * ((b - 24) / 8) ** 2
    if b < 48:
        return 5200
    if b < 64:
        return 2300
    if b < 72:
        return 1200 + 5500 * ((b - 64) / 8) ** 2
    if b < 88:
        return 6500
    return max(3000 * (1 - (b - 88) / 8) + 500, 500)


def build_pad():
    pad = stereo()
    for bar in range(NBARS):
        root, ivs = parse_chord(BAR_CHORDS[bar])
        notes = [voicing(root, [0], 43)[0]] + voicing(root, ivs, 55)
        sec = BAR_SECTION[bar]
        g = {"intro": 0.8, "verse1": 0.6, "pre": 0.75, "chorus1": 0.9, "verse2": 0.6,
             "bridge": 0.85, "chorus2": 0.95, "outro": 0.8}[sec]
        dur = BAR + 1.2
        x = pad_chord(notes, dur, seed=bar)
        n = x.shape[1]
        e = env_adsr(n, a=0.25 if sec in ("intro", "bridge", "outro") else 0.04, d=1.0, s=0.85, r=1.1, gate=BAR)
        place(pad, x * e[None, :], bar_t(bar), g)
    pad = tv_filter(pad, pad_cutoff, "low", order=2)
    return pad


def build_arp():
    arp = stereo()
    pat = [0, 2, 4, 1, 3, 5, 2, 4, 0, 2, 4, 1, 5, 3, 4, 2]
    for bar in range(NBARS):
        sec = BAR_SECTION[bar]
        rel = bar - SEC_START[sec]
        on = {"intro": rel >= 2, "verse1": rel >= 8, "pre": True, "chorus1": rel < 14,
              "verse2": True, "bridge": True, "chorus2": rel < 14, "outro": True}[sec]
        if not on:
            continue
        root, ivs = parse_chord(BAR_CHORDS[bar])
        tones = voicing(root, ivs, 64)
        tones = tones + [x + 12 for x in tones]
        bright = {"intro": 0.5 + 0.1 * rel, "verse1": 0.6, "pre": 0.7 + 0.08 * rel, "chorus1": 1.3,
                  "verse2": 0.8, "bridge": 0.7 + 0.1 * rel, "chorus2": 1.5, "outro": 0.9 - 0.08 * rel}[sec]
        g = {"intro": 0.5 + 0.08 * rel, "outro": max(0.9 - 0.1 * rel, 0.2)}.get(sec, 0.9)
        for i in range(16):
            m = tones[pat[i]]
            place(arp, pluck(m, 0.45, max(bright, 0.3)), bar_t(bar, i * 0.25),
                  g * (1.0 if i % 4 == 0 else 0.7), 0.35 * np.sin(i * 0.9))
    arp = arp + pingpong(arp, 3 * STEP, 0.42, 5) * 0.45
    return arp


def build_lead(robot_events):
    """Saw lead doubling the robot melody an octave up in the final chorus."""
    lead = stereo()
    prev = None
    for ev in robot_events:
        if ev["kind"] != "chunk" or ev["t0"] < SEC_START["chorus2"] * BAR:
            continue
        m = ev["note"] + 12
        x = lead_note(m, ev["slot"] * 0.9, prev)
        place(lead, x, ev["t0"], 0.9, -0.25)
        place(lead, x, ev["t0"] + 0.012, 0.6, 0.35)
        prev = m
    return lead


def build_fx():
    fx = stereo()
    boom = impact()
    for t in (16.0, 64.0, 144.0):
        place(fx, boom, t, 1.0)
        place(fx, downlifter(3.0, seed=int(t)), t, 0.35)
    for t in (96.0,):
        place(fx, downlifter(3.0, seed=11), t, 0.4)
    # risers end exactly on the drop
    place(fx, riser(6.0, seed=1), 16.0 - 6.0, 0.45)
    place(fx, riser(8.0, seed=2), 64.0 - 8.0, 0.55)
    place(fx, riser(8.0, seed=4), 144.0 - 8.0, 0.55)
    for t in (16.0, 64.0, 144.0, 96.0):
        place(fx, reverse_cymbal(2.0), t - 2.0, 0.35)
    # intro "wind": filtered noise swelling up from silence
    rng = np.random.default_rng(12)
    n = int(12 * SR)
    wind = tv_filter(rng.standard_normal((2, n)), lambda tt: 400 + 600 * np.sin(tt * 0.7) ** 2, "band")
    tt = np.arange(n) / SR
    place(fx, wind * (np.clip(tt / 4, 0, 1) * np.clip((12 - tt) / 4, 0, 1))[None, :], 0.0, 0.12)
    return fx


# --------------------------------------------------------------------------
# Vocals
# --------------------------------------------------------------------------
def chord_at(t):
    bar = min(int(t / BAR), NBARS - 1)
    return parse_chord(BAR_CHORDS[bar])


def show_tokens(show):
    toks = []
    for raw in show.split():
        hl = "*" in raw
        toks.append({"w": raw.replace("*", ""), "hl": hl})
    return toks


def map_words(say, show, words, t0):
    """Assign a start/end time to every displayed word."""
    say_words = say.split()
    toks = show_tokens(show)
    if len(words) != len(say_words):
        # fall back to spreading by character count
        total = sum(len(w) for w in say_words)
        span = words[-1][1] if words else 1.0
        acc, words = 0, []
        for w in say_words:
            a = acc / total * span
            acc += len(w)
            words.append((a, acc / total * span))
    # character timeline of the spoken text
    cpos, cum = [], 0
    for w, (a, b) in zip(say_words, words):
        cpos.append((cum, cum + len(w), a, b))
        cum += len(w) + 1
    say_len = max(cum - 1, 1)
    show_len = max(sum(len(t["w"]) + 1 for t in toks) - 1, 1)

    def time_at(frac):
        c = frac * say_len
        for c0, c1, a, b in cpos:
            if c <= c1:
                return a + (b - a) * np.clip((c - c0) / max(c1 - c0, 1), 0, 1)
        return cpos[-1][3]

    out, cum = [], 0
    for tk in toks:
        a = time_at(cum / show_len)
        b = time_at((cum + len(tk["w"])) / show_len)
        cum += len(tk["w"]) + 1
        out.append({"w": tk["w"], "hl": tk["hl"], "t0": round(t0 + a, 3), "t1": round(t0 + max(b, a + 0.05), 3)})
    return out


def build_narration():
    stem = stereo()
    events = []
    for ln in song["narration"]:
        audio, words = tts("narrator", ln["say"])
        nat = len(audio) / SR
        ls = float(np.clip(ln["max"] * 0.97 / nat, 0.86, 1.1))
        if abs(ls - 1.0) > 0.01:
            audio, words = tts("narrator", ln["say"], length_scale=ls)
        dur = len(audio) / SR
        if dur > ln["max"] + 0.05:
            print(f"WARNING narration at {ln['t']} is {dur:.2f}s > {ln['max']}s", file=sys.stderr)
        t0 = ln["t"] + 0.04
        x = audio / (np.abs(audio).max() + 1e-9)
        place(stem, fade(x, 0.005, 0.03), t0, 1.0)
        events.append({
            "t0": round(t0, 3), "t1": round(t0 + dur, 3), "style": ln.get("style", "line"),
            "show": ln["show"].replace("*", ""), "words": map_words(ln["say"], ln["show"], words, t0),
        })
    # voice chain: HPF, presence, gentle compression
    stem = butter(stem, 90, "high")
    stem = peaking(stem, 3200, 2.5, 0.8)
    stem = peaking(stem, 250, -2.0, 1.0)
    env = smooth_env(stem.mean(0), 12)
    thr = np.percentile(env[env > 1e-4], 70)
    g = np.where(env > thr, (thr / np.maximum(env, 1e-9)) ** 0.4, 1.0)
    stem = stem * g[None, :]
    return stem, events


def carrier_notes(note, t0, slot):
    root, ivs = chord_at(t0 + 0.01)
    chord = voicing(root, ivs, 48)

    def fn(t):
        vib = 1 + 0.005 * np.sin(2 * np.pi * 5.6 * t) * np.clip((t - 0.18) / 0.25, 0, 1)
        parts = [(mtof(note) * vib, 1.0), (mtof(note - 12) * vib, 0.45)]
        parts += [(mtof(c), 0.22) for c in chord]
        return parts
    return fn


def sing_world(audio, target_len, note, frame_ms=5.0):
    """Speech-to-singing with the WORLD vocoder: keep the spectral envelope,
    replace the pitch with the melody note and stretch the vowels to fit."""
    import pyworld as pw
    x = np.ascontiguousarray(audio, dtype=np.float64)
    f0, tt = pw.harvest(x, SR, frame_period=frame_ms, f0_floor=60, f0_ceil=400)
    sp = pw.cheaptrick(x, f0, tt, SR)
    ap = pw.d4c(x, f0, tt, SR)
    T = len(f0)
    voiced = (f0 > 0).astype(float)
    To = max(int(target_len * 1000 / frame_ms), 2)
    ratio = To / T
    w = 1 + (To - T) / max(voiced.sum(), 1) * voiced if ratio > 1 else np.full(T, ratio)
    cum = np.concatenate([[0], np.cumsum(w)])
    cum *= To / cum[-1]
    idx = np.clip(np.round(np.interp(np.arange(To), cum[:-1], np.arange(T))).astype(int), 0, T - 1)
    t = np.arange(To) * frame_ms / 1000
    fn = mtof(note) * (1 + 0.006 * np.sin(2 * np.pi * 5.6 * t) * np.clip((t - 0.18) / 0.25, 0, 1))
    y = pw.synthesize(np.where(voiced[idx] > 0, fn, 0.0), np.ascontiguousarray(sp[idx]),
                      np.ascontiguousarray(ap[idx]), SR, frame_ms)
    return fade(y / (np.abs(y).max() + 1e-9), 0.003, 0.03)


def split_phrase(says, length_scale=0.95):
    """Speak a whole phrase (natural coarticulation), then cut it into chunks
    at the word boundaries reported by the TTS alignment."""
    audio, words = tts("robot", ", ".join(says) + "!", length_scale)
    counts = [len(s.split()) for s in says]
    if len(words) != sum(counts):
        print(f"WARNING alignment mismatch for {says}; synthesizing chunks separately", file=sys.stderr)
        return [tts("robot", s + ".", length_scale)[0] for s in says]
    starts, k = [], 0
    for c in counts:
        starts.append(words[k][0])
        k += c
    pieces = []
    for i, st in enumerate(starts):
        a = int(max(st - 0.015, 0) * SR)
        b = int((starts[i + 1] - 0.015) * SR) if i + 1 < len(starts) else len(audio)
        p = audio[a:b]
        env = uniform_filter1d(np.abs(p), 64)
        idx = np.where(env > 0.01)[0]
        if len(idx):
            p = p[: idx[-1] + 300]
        pieces.append(fade(p.copy(), 0.002, 0.01))
    return pieces


def build_robot():
    stem = stereo()
    events = []
    groups = []  # each group: list of chunk dicts spoken as one phrase
    letters = [("J", "J"), ("E", "E"), ("P", "P"), ("A", "A")]
    for L in song["robot"]["letters"]:
        groups.append([dict(t0=L["t"] + i * BEAT, slot=BEAT, note=L["notes"][i], say=say, show=show,
                            kind="letter", group="letters", idx=i) for i, (say, show) in enumerate(letters)])
    for ch in song["robot"]["choruses"]:
        for p, pname in enumerate(ch["phrases"]):
            ph = song["robot"]["phrases"][pname]
            chunks = letters if pname == "jepa" else ph["chunks"]
            groups.append([dict(t0=ch["t"] + p * 2 * BAR + i * 2 * BEAT, slot=2 * BEAT,
                                note=note + ch["transpose"], say=say, show=show,
                                kind="letter" if pname == "jepa" else "chunk", group=pname, phrase=p, idx=i)
                           for i, ((say, show), note) in enumerate(zip(chunks, ph["notes"]))])
    for e in song["robot"]["echo"]:
        groups.append([dict(t0=e["t"], slot=e["slot"], note=e["note"], say=e["say"], show=e["show"],
                            kind="echo", group="echo", idx=0)])
    k = 0
    for grp in groups:
        pieces = split_phrase([it["say"] for it in grp])
        for it, audio in zip(grp, pieces):
            nat = len(audio) / SR
            target = max(min(it["slot"] * 0.86, nat * 3.0), min(nat, it["slot"] * 0.95))
            fn = carrier_notes(it["note"], it["t0"], it["slot"])
            left = vocode(audio, target, fn, seed=k, detune=-8)
            right = vocode(audio, target, fn, seed=k + 1000, detune=8)
            sung = sing_world(audio, target, it["note"])
            n = min(len(left), len(sung))
            center = sung[:n] * 0.8
            x = np.stack([0.55 * left[:n] + center, 0.55 * right[:n] + center])
            place(stem, x, it["t0"], 1.0)
            ev = {k2: it[k2] for k2 in ("kind", "group", "show", "note", "slot", "idx")}
            ev.update(t0=round(it["t0"], 3), t1=round(it["t0"] + target, 3))
            if "phrase" in it:
                ev["phrase"] = it["phrase"]
            events.append(ev)
            k += 1
    stem = butter(stem, 140, "high")
    stem = peaking(stem, 2500, 3.0, 0.9)
    return stem, sorted(events, key=lambda e: e["t0"])


# --------------------------------------------------------------------------
# Mix & master
# --------------------------------------------------------------------------
def sidechain(times, depth=0.6, release=0.16, n=N + TAIL):
    env = np.ones(n)
    L = int(0.4 * SR)
    shape = 1 - depth * np.exp(-np.arange(L) / (release * SR))
    shape[: int(0.004 * SR)] = np.linspace(1, shape[int(0.004 * SR)], int(0.004 * SR))
    for t in times:
        i = int(t * SR)
        j = min(i + L, n)
        env[i:j] = np.minimum(env[i:j], shape[: j - i])
    return env


def limiter(x, ceiling=0.93, look=0.004, release=0.08):
    peak = maximum_filter1d(np.abs(x).max(0), int(look * SR) * 2 + 1)
    g = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    g = minimum_filter1d(g, int(release * SR))
    g = uniform_filter1d(g, int(look * SR) * 2 + 1)
    return np.clip(x * g[None, :], -ceiling, ceiling)


def write_wav(path, x):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    y = np.clip(x, -1, 1)
    pcm = (y.T * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main():
    print("drums ...", file=sys.stderr)
    drums, kick_times = build_drums()
    print("bass/pad/arp ...", file=sys.stderr)
    bass, pad, arp = build_bass(), build_pad(), build_arp()
    print("fx ...", file=sys.stderr)
    fx = build_fx()
    print("narration ...", file=sys.stderr)
    narr, narr_events = build_narration()
    print("robot vocoder ...", file=sys.stderr)
    robot, robot_events = build_robot()
    lead = build_lead(robot_events)

    stems = dict(drums, bass=bass, pad=pad, arp=arp, fx=fx, lead=lead, narration=narr, robot=robot)
    targets = dict(kick=-15, snare=-23, hats=-29, crash=-28, bass=-17.5, pad=-22, arp=-25,
                   fx=-25, lead=-27, narration=-10.5, robot=-11.5)
    for k, v in stems.items():
        g = 10 ** ((targets[k] - active_rms_db(v)) / 20)
        stems[k] = v * g

    # sidechain pumping from the kick, ducking the music under the narration
    sc = sidechain(kick_times, 0.55)
    for k in ("pad", "arp", "lead"):
        stems[k] = stems[k] * sc[None, :]
    stems["bass"] = stems["bass"] * sidechain(kick_times, 0.35, 0.1)[None, :]
    vox = smooth_env(stems["narration"].mean(0), 6)
    duck = 1 - 0.4 * np.clip(vox / (np.percentile(vox[vox > 1e-4], 60) + 1e-9), 0, 1)
    rvox = smooth_env(stems["robot"].mean(0), 6)
    duck *= 1 - 0.3 * np.clip(rvox / (np.percentile(rvox[rvox > 1e-4], 60) + 1e-9), 0, 1)
    for k in ("pad", "arp", "lead", "bass", "hats", "crash", "fx"):
        stems[k] = stems[k] * duck[None, :]

    # effect sends
    hall = make_ir(2.8, seed=21)
    plate = make_ir(1.4, seed=22, damp=0.3)
    send_hall = stems["pad"] * 0.25 + stems["arp"] * 0.3 + stems["fx"] * 0.5 + stems["lead"] * 0.3 + stems["robot"] * 0.22
    send_plate = stems["snare"] * 0.25 + stems["narration"] * 0.13 + stems["robot"] * 0.12
    wet = reverb(butter(send_hall, 250, "high"), hall) * 0.5 + reverb(butter(send_plate, 300, "high"), plate) * 0.45
    robot_delay = pingpong(stems["robot"], 3 * STEP, 0.35, 4, 3500) * 0.28

    mix = sum(stems.values()) + wet + robot_delay
    mix = butter(mix, 28, "high")
    # final fade
    n_end = N
    fo = int(5.0 * SR)
    mix = mix[:, :n_end]
    mix[:, n_end - fo:] *= np.linspace(1, 0, fo) ** 1.5

    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(SR)
        for _ in range(3):
            lufs = meter.integrated_loudness(limiter(mix).T)
            mix *= 10 ** ((-14.0 - lufs) / 20)
        master = limiter(mix)
        print(f"integrated loudness: {meter.integrated_loudness(master.T):.1f} LUFS", file=sys.stderr)
    except ImportError:
        master = limiter(mix / (np.abs(mix).max() + 1e-9) * 1.6)
    print(f"peak: {20*np.log10(np.abs(master).max()):.2f} dBFS", file=sys.stderr)

    write_wav(os.path.join(BUILD, "soundtrack.wav"), master)
    for k, v in stems.items():
        write_wav(os.path.join(BUILD, "stems", f"{k}.wav"), v[:, :N] / max(1.0, np.abs(v).max()))

    # energy envelopes for audio-reactive visuals (30 fps)
    fps = 30
    hop = SR // fps
    mono = master.mean(0)
    nfr = int(DUR * fps)
    lvl = np.array([np.sqrt(np.mean(mono[i * hop:(i + 1) * hop] ** 2)) for i in range(nfr)])
    lowb = butter(mono, 150)
    low = np.array([np.sqrt(np.mean(lowb[i * hop:(i + 1) * hop] ** 2)) for i in range(nfr)])
    lvl /= lvl.max() + 1e-9
    low /= low.max() + 1e-9

    timing = {
        "title": song["title"], "bpm": BPM, "duration": DUR,
        "sections": [{"name": s["name"], "t0": s["bar"] * BAR, "t1": (s["bar"] + s["bars"]) * BAR} for s in song["sections"]],
        "chords": BAR_CHORDS,
        "kicks": [round(t, 3) for t in kick_times],
        "narration": narr_events,
        "robot": robot_events,
        "env": {"fps": fps, "level": [round(float(v), 3) for v in lvl], "low": [round(float(v), 3) for v in low]},
    }
    with open(os.path.join(ROOT, "mv", "timing.js"), "w") as f:
        f.write("// Generated by src/compose.py — do not edit by hand.\n")
        f.write("window.TIMING = " + json.dumps(timing, separators=(",", ":")) + ";\n")
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
