# J.E.P.A. — *Predict What Matters*

A 3‑minute music video that introduces **JEPA (Joint Embedding Predictive Architecture)**, the
self‑supervised learning idea Yann LeCun proposed in *A Path Towards Autonomous Machine
Intelligence* (2022), and follows it through I‑JEPA, V‑JEPA and V‑JEPA 2.

Two cuts are rendered from the same source:

| Version | File | Resolution |
| --- | --- | --- |
| Landscape 16:9 (YouTube, desktop) | [`output/JEPA-MV_16x9.mp4`](output/JEPA-MV_16x9.mp4) | 1920 × 1080, 30 fps |
| Portrait 9:16 (Shorts, Reels, TikTok) | [`output/JEPA-MV_9x16.mp4`](output/JEPA-MV_9x16.mp4) | 1080 × 1920, 30 fps |

Everything (music, vocals and animation) is generated from code in this repository. No
stock footage, samples or recordings are used.

## What the video covers

| Time | Section | Content |
| --- | --- | --- |
| 0:00 | Prologue | LeCun's opening question: *"How could machines learn as efficiently as humans and animals?"* |
| 0:16 | The problem with pixels | A baby learns by watching. Generative models try to paint every pixel of the future, and because the future is uncertain, all the possible futures blur together. |
| 0:48 | The idea | Don't predict pixels. Encode the world into an abstract representation space and make predictions there. |
| 1:04 | The architecture (chorus) | The context encoder turns *x* into *s_x*, the target encoder turns *y* into *s_y*, and the predictor (with latent *z*) outputs ŝ_y. The energy is *E(x, y) = D(s_y, Pred(s_x, z))*. |
| 1:36 | Avoiding collapse | If every input maps to one point, the energy is zero everywhere and the model learns nothing. Two fixes: VICReg‑style variance/covariance regularization, or an EMA target encoder with stop‑gradient. |
| 1:52 | I‑JEPA | From one large context block, predict the *features* of several target blocks. No hand‑crafted augmentations and no pixel decoder. ViT‑H/14 on ImageNet trains on 16 A100s in under 72 hours. |
| 2:08 | V‑JEPA | Masked space‑time tubes in video. Intuitive physics shows up as a spike in "surprise" when something impossible happens. |
| 2:16 | World models | V‑JEPA 2 imagines futures in latent space and picks the plan with the lowest energy to reach the goal. It needs less than 62 hours of robot video. |
| 2:24 | The big picture (chorus) | The full architecture, then H‑JEPA (a hierarchy of short‑term and long‑term predictions), then a timeline from 2022 to 2025. |
| 2:56 | Epilogue | *"Not every pixel. Not every detail. Predict what matters."* Ends with a further‑reading card. |

## How it is made

```
src/song.json       lyrics, chords, structure and timing (the single source of truth)
src/compose.py      synthesizes the soundtrack and writes mv/timing.js
mv/index.html       interactive player (toggle 16:9 / 9:16, scrub, play with audio)
mv/mv.js            the renderer: MV.renderFrame(t) draws any frame as a pure function of time
mv/fonts/           Space Grotesk, Inter, JetBrains Mono (SIL OFL)
scripts/render.mjs  headless-Chromium frame capture → H.264 segments → MP4 with audio
scripts/snapshot.mjs  render individual frames to PNG for review
```

**Music.** `compose.py` builds a 120 BPM synthwave track in A minor and lifts the final chorus to
B minor. Every sound is synthesized with numpy and scipy: the kick, snare, clap, hats and crash;
an additive‑synthesis bass and arp; supersaw pads; and risers and impacts. The mix uses
sidechain pumping, vocal ducking, convolution reverb, ping‑pong delay and a look‑ahead
limiter, and the master is normalized to −14 LUFS.

**Vocals.** The verses are spoken by a [Piper](https://github.com/OHF-Voice/piper1-gpl) TTS
voice (`en_US-lessac-high`). Piper's phoneme alignments give word‑level timestamps, which drive
the karaoke‑style lyrics. The chorus robot is a second Piper voice (`en_US-ryan-high`). Each
phrase is spoken as a whole, cut at word boundaries, and turned into singing in two ways: a
WORLD‑vocoder resynthesis that replaces the pitch with the melody note and stretches the vowels
to fit the beat, and a 40‑band channel vocoder that plays the chord, used for the stereo robot
layers. Intelligibility was checked by transcribing the stems and the final mix with Whisper.

**Visuals.** `mv.js` draws each frame on a 2D canvas: scene logic, kinetic typography, a bloom
pass and beat‑reactive accents driven by the kick times and energy envelope exported from the
audio. Layouts are defined for both orientations. In portrait, diagrams are re‑flowed vertically
and captions stay clear of the areas that platform UI usually covers.

## Rebuilding

Requirements: Python 3.10+, Node 18+, ffmpeg with libx264, and Chromium for Playwright.

```bash
pip install numpy scipy piper-tts onnx pyworld pyloudnorm
npm install                     # playwright
npx playwright install chromium # if you don't already have it

python3 src/compose.py          # → build/soundtrack.wav, mv/timing.js (downloads the Piper voices on first run)
ffmpeg -i build/soundtrack.wav -c:a aac -b:a 192k mv/soundtrack.m4a   # audio for the web player
node scripts/render.mjs 16x9    # → output/JEPA-MV_16x9.mp4
node scripts/render.mjs 9x16    # → output/JEPA-MV_9x16.mp4
```

To preview interactively, serve the folder (for example `npx http-server .`) and open
`mv/index.html`. Use the buttons to switch between 16:9 and 9:16.

To change a lyric, edit `src/song.json` and re‑run `compose.py`. The vocal timings and on‑screen
words update automatically.

## Further reading

- Y. LeCun, *A Path Towards Autonomous Machine Intelligence*, 2022.
- M. Assran et al., *Self‑Supervised Learning from Images with a Joint‑Embedding Predictive Architecture* (I‑JEPA), CVPR 2023.
- A. Bardes et al., *Revisiting Feature Prediction for Learning Visual Representations from Video* (V‑JEPA), 2024.
- M. Assran et al., *V‑JEPA 2: Self‑Supervised Video Models Enable Understanding, Prediction and Planning*, 2025.
- A. Bardes, J. Ponce, Y. LeCun, *VICReg: Variance‑Invariance‑Covariance Regularization for Self‑Supervised Learning*, ICLR 2022.

---

This is a fan‑made educational video. It is not affiliated with or endorsed by Meta or Yann
LeCun. The fonts are licensed under the SIL Open Font License (see `mv/fonts/LICENSE-*`).

**Voice licensing:** both Piper voices were trained on datasets licensed for non‑commercial use
only: `lessac` uses the [Blizzard 2013 Lessac license](https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html)
and `ryan` uses RyanSpeech (CC BY‑NC‑SA 4.0). The rendered videos are therefore suitable for
educational and non‑commercial sharing. For commercial use, swap in voices with a commercial
license by changing `VOICES` in `src/compose.py`.
