# Timestamping Audio with Electric Network Frequency

Recovering *when* a recording was made from a hum nobody meant to record.

## Why this is possible

The European grid runs at a nominal 50 Hz, but it never sits exactly there. The
frequency is set by the instantaneous balance between generation and load: when
demand outruns supply the rotating mass of every generator on the grid slows
fractionally, when supply outruns demand it speeds up. Operators correct
continuously, so the frequency wanders around 50 Hz within a band of roughly
±0.05 Hz.

Three properties turn that wandering into a clock:

- **It is shared.** A synchronous area moves as one machine. At any instant the
  frequency in Warsaw is the frequency in Lisbon, so a single reference series
  covers a continent.
- **It is unpredictable.** The deviation sequence depends on aggregate human
  behaviour and cannot be modelled forward, which makes any given stretch of it
  effectively unique.
- **It is recorded.** Grid operators log frequency continuously, so a reference
  history exists to match against.

A recording that captured the grid frequency therefore carries a timestamp, and
one that is hard to forge — you would have to synthesise a deviation sequence
matching the real grid's history for the moment you were claiming. This is used
in forensic audio authentication, where the question is whether a piece of
evidence was recorded when its owner says it was.

## How the hum gets into the recording

Not by one route, which matters for extraction because the routes land in
different places in the spectrum:

**Conducted.** A mains-powered recorder never fully rejects its own supply. The
ripple that survives the power supply leaks into the audio path at the line
frequency. For mains-powered equipment this is usually the strongest path.

**Inductive.** Mains wiring, transformers and motors radiate at the line
frequency; any loop of wire nearby — an unshielded cable, a microphone's voice
coil — has a current induced in it. This is ordinary mains hum, and it is the
path most people picture, though it is often not the dominant one.

**Acoustic.** Mains-powered hardware physically vibrates, and this component
arrives at **twice** the line frequency: magnetostrictive force follows the
square of the magnetic field, so a 50 Hz field pushes and pulls a core 100 times
a second. A battery-powered recorder in a room with a transformer can still
capture ENF this way.

The consequence for extraction is that the signal may be strongest at 50, 100 or
150 Hz depending on the recording setup, and the harmonic with the best
signal-to-noise ratio is not always the fundamental.

## What this repository does

| Notebook | Role |
|---|---|
| `precomp.ipynb` | extracts the ENF trace from an audio file |
| `precomp-demo.ipynb` | the same pipeline on the bundled sample, step by step |
| `timestamping.ipynb` | matches an extracted trace against a reference series |

## Extraction

A short-time Fourier transform trades frequency resolution against time
resolution, and the numbers here are unforgiving. The transform uses a 10-second
Gaussian window, which puts bin spacing at 0.1 Hz — wider than the entire
±0.05 Hz band the signal moves in. Bin position alone therefore carries almost
no information, and the frequency has to be read from where the peak sits
*between* bins.

Time resolution is bought back with overlap rather than with a shorter window:
the hop is 0.5 s, so consecutive windows share 95% of their samples and the
trace gets a point every half second while each point still integrates over ten.
A Gaussian window suits this because it minimises the time-bandwidth product —
no window shape gives a tighter peak for a given length.

```python
w = windows.gaussian(fs * 10, std=(fs * 10) // 6)
SFT = ShortTimeFFT(win=w, hop=fs // 2, fs=fs, scale_to="magnitude")
```

What comes out is an image in which the ENF is a faint horizontal ridge, broken
wherever noise wins. Recovering a continuous trace from it is posed as a
clustering problem rather than a curve-fitting one:

1. **Treat the spectrogram as a point cloud** — pixels brighter than 10% above
   the mean become points, with magnitude as the third dimension.
2. **Density-based cleaning.** `DBSCAN(eps=11, min_samples=100)` over the
   time-frequency coordinates, then five passes of IQR trimming. This works
   because of how the noise is shaped here: it arrives as thin bands rather than
   as a diffuse floor, and a thin band is sparse in its own neighbourhood — so
   density is exactly the property that separates it from the ridge.
3. **Average per time bin**, collapsing each surviving column to one frequency.
4. **Interpolate the gaps** left where noise had taken a column out entirely.
5. **Exponential smoothing** at level 0.08. The raw trace jumps enough that
   without it the matching step has no stable shape to align.

This is deliberately not the state of the art. Dedicated ridge-tracking
algorithms exist, and given ground-truth ENF values paired with spectrograms the
problem suits a convolutional network. The constraint was to build it from
methods covered in an introductory machine learning course — and density-based
clustering turns out to fit the geometry of a broken ridge better than that
constraint suggests it should.

## Matching

Dating reduces to alignment: slide the extracted trace along the reference
series, score the disagreement at every offset, and take the minimum. A correct
match shows a sharp, isolated minimum; a recording with no usable ENF produces a
flat scoring curve with no winner, which is itself a useful answer.

**Dynamic time warping was tried and abandoned.** It did not merely rank the
correct offset poorly — across recordings the true timestamp was essentially
never among the k best candidates, and the candidates it did return had shapes
visibly unlike the query. A method that wrong is not mistuned.

The likely reason is that warping is the wrong prior for this problem. DTW earns
its keep when two sequences describe the same events at different or varying
speeds. Here they do not: the recorder's clock and the grid's clock both advance
in real seconds, so the correct correspondence is a rigid shift. Given freedom to
stretch, the algorithm can fit a query to almost any stretch of reference, which
is exactly how unrelated segments end up scoring better than the right one.

## What could and could not be measured

The recordings available here have no published ground-truth ENF series, so
there was no way to score extraction numerically — no error in millihertz, no
detection rate. Parameters were judged visually, against 2D spectrograms and 3D
surfaces of the time-frequency plane, and the pipeline is reported as working
because the traces it produces are continuous, plausible and match the reference
at a consistent offset.

That is a real limitation and it bounds every claim on this page. A dataset
pairing recordings with logged grid frequency would turn the visual judgement
into a measurable one, and would also unlock the supervised approaches this
project deliberately avoided.

## Not in this repository

One further piece of the work is **not** here, and is mentioned so that nothing
on this page is taken to cover it: characterising and classifying different
synchronous grids from their frequency behaviour. That meant feature
engineering over historical series — variance and the density of stationary
points in the derivative gave the widest class separation — with t-SNE used to
inspect the feature space and LightGBM to classify, benchmarked against
logistic regression and random forests, all of which cleared a high baseline.

None of that code, and neither t-SNE nor LightGBM, appears in this repository.
What is here is extraction and timestamping.

## Data

The bundled sample is from the [ENF-WHU
dataset](https://github.com/ghua-ac/ENF-WHU-Dataset), MIT licensed, © 2023 HUA
GUANG.
