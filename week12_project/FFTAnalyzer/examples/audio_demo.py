"""
audio_demo.py
----------------
Audio processing demo. Run from the FFTAnalyzer/ folder:

    python3 examples/audio_demo.py                 # uses a synthetic clip
    python3 examples/audio_demo.py my_clip.wav     # or your own WAV file

With no argument it builds a synthetic "recording" (three harmonic
notes + 60 Hz mains hum + white noise at 15 dB SNR), saves it, then
cleans it up: notch the hum and its harmonics, low-pass at 2 kHz,
spectral-subtraction denoise. Because the clean signal is known, it
reports the SNR after each step.

Writes to examples/output/: audio_noisy.wav, audio_cleaned.wav,
audio_spectrograms.png, audio_spectra.png.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fft_analyzer.applications.audio_processor import AudioProcessor, make_demo_clip, write_wav
from fft_analyzer.signal.generators import snr_db

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main(path=None):
    os.makedirs(OUT, exist_ok=True)
    clean = None
    if path is None:
        clean, noisy, fs = make_demo_clip()
        path = os.path.join(OUT, "audio_noisy.wav")
        write_wav(path, noisy, fs)
        print("wrote synthetic noisy clip:", path)

    ap = AudioProcessor.load(path)
    print(f"loaded {path}: {ap.channels} channel(s), {ap.sample_rate} Hz, {ap.duration:.2f} s")
    print("dominant frequencies before:",
          [(round(f, 1), round(a, 3)) for f, a in ap.dominant_frequencies(6)])

    def report(step):
        if clean is not None:
            print(f"  after {step:28s} SNR = {snr_db(clean, ap.mono()):6.2f} dB")

    report("loading (16-bit WAV)")
    ap.remove_hum(60, harmonics=3)
    report("hum notches 60/120/180 Hz")
    print(f"  60 Hz band energy change: {ap.band_energy_change(55, 65):+.1f} dB")
    ap.low_pass(2000, transition=100)
    report("low-pass 2 kHz")
    ap.spectral_denoise()
    report("spectral subtraction")
    ap.normalize(0.9)

    out = ap.save(os.path.join(OUT, "audio_cleaned.wav"))
    print("wrote", out)
    plt.close(ap.plot_spectrogram(os.path.join(OUT, "audio_spectrograms.png"), fmax=2500))
    plt.close(ap.plot_spectra(os.path.join(OUT, "audio_spectra.png"), xlim=(0, 3000)))
    print("figures in", OUT)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
