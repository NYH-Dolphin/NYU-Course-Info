# Lecture8 Audio Understanding

## 1. Basic of Audio and Sound

### Audio Signal / Waveform

Change in air pressure at a specific point in space over time registered by a microphone.

<img src="./Lecture8 Audio Understanding.assets/image-20240409125923842.png" alt="image-20240409125923842" style="zoom:67%;" />

### Frequency and Pitch

- A signal is **periodic** if it repeats
- Formally, a signal $x[n]$ is periodic if there is a such that $x[n]=x[n+p]$ for all $n$. $p$​ is the period of the signal, typically measured in seconds
- **Frequency** is the reciprocal (倒数) of the **period**, $f=\frac{1}{p}$​,  typically measured in Hertz (Hz) (cycles per second)

#### Elements

<img src="./Lecture8 Audio Understanding.assets/image-20240409130125118.png" alt="image-20240409130125118" style="zoom:67%;" />

A sinusoid is the simplest type of periodic waveform, fully described by its:

- **frequency** $f$ (number of cycles per second)
- **amplitude** $a$ (peak deviation from the mean)
- **phase** $\phi$​ (where in the cycle is the sinusoid at time zero)

#### Music Frequency

<img src="./Lecture8 Audio Understanding.assets/image-20240409130322719.png" alt="image-20240409130322719" style="zoom:50%;" />

- “Real” sounds are more complex than pure tones
- Musical tones can be modeled as a combination of different pure tones (**partials**), each with different frequencies, amplitudes, and phases
- The frequency of the lowest partial is the **fundamental frequency (f0)**, which often  determines the perceived pitch
- The frequencies of the other partials which are an integer number of the f0 are called **harmonics**

#### Real World Frequency

- The larger the **frequency** of a sinusoidal waveform, the “**higher**” it sounds
- Human hearing range: 20 Hz — 20 kHz
    - Orchestra tuning frequency: 440 Hz
- **Frequency** is closely related to **pitch**, which is subjective
    - For pure tones, we can consider pitch and frequency equal

#### Pitch

- Pitch is the human perception of the frequency of a sound wave.
- It's a **subjective** quality that can be described as "how high" or "how low" a sound seems to the human ear.

### Loudness and Decibel

- **Loudness**: a perceived measure of intensity of a sound, correlated with the objective measures of **sound intensity** and **sound power**

#### Decibel

- **Decibel** (dB): a scale, measuring the **logarithmic** ratio between a **sound’s intensity** and the **threshold of hearing (the most silent sound human can notice)**



<img src="./Lecture8 Audio Understanding.assets/image-20240409130828889.png" alt="image-20240409130828889" style="zoom:67%;" />

- **Intensity**: A physical measure of the power per unit area, expressed in watts per square meter.
- **Intensity level**: The intensity of the sound converted into decibels (dB). This is a logarithmic scale used to describe the intensity level in a way that is more manageable for human perception.
- **× TOH**: This column shows how many times the intensity of each sound source is greater than the threshold of hearing. The threshold of hearing is typically taken as $10^{-12}$​ watts per square meter and is equivalent to 0 dB.

#### Phon Level (Loudness)

<img src="./Lecture8 Audio Understanding.assets/image-20240409130937113.png" alt="image-20240409130937113" style="zoom:67%;" />

- The 'phon' is a unit that is used to describe **loudness** levels. Equal loudness contours show curves with a **constant perceived loudness** (in units of  “phons”)
- The **perceived** quantity, **loudness**, depends  on both **intensity** and **frequency**
- Our ears are most sensitive between **2-4kHz**, which is where the curves dip the lowest. This means it takes **less intensity** (fewer decibels) for sounds in this frequency range to be perceived as **loud**.
- The intensity of the threshold of hearing/ pain depends on the frequency

### Timbre (音色)

- The perceptual property of sound that allows us to **distinguish between a instruments** if they played the same note at the same volume
- A perceived quality of sound, related to
    - The evolution of the sound over time
    - The distribution of energy across partials
    - The relative amount of noise and sinusoidal components

### Fourier transform

Waveform not the most informative to look at. There are other tools that make the spectral content of the audio more **explicit**.

Which frequencies are present in an audio signal?

1. Get a set of “sinusoids” with the frequencies of interest
2. For each “sinusoid”, compare to original signal: how well do they match?
3. You can recover the signal by adding up those sinusoids

<img src="./Lecture8 Audio Understanding.assets/image-20240409133358213.png" alt="image-20240409133358213" style="zoom:67%;" />

- (c) matches the best

#### Discrete Fourier transformation (DFT)

The DFT (discrete Fourier transformation) shows the overall frequency composition of a sound, but we lose information about changes over time

<img src="./Lecture8 Audio Understanding.assets/image-20240409134022424.png" alt="image-20240409134022424" style="zoom:67%;" />

- **Time Domain Signals** (left)
    - Shows a simple periodic signal, likely a pure sine wave, followed by a segment of higher frequency oscillations. This could represent a sound that changes pitch over time
- **Frequency Domain Representations** (right)
    - The corresponding frequency domain graph, as analyzed by the DFT.
    - These graphs show how much of each frequency is present in the original time domain signal. Peaks in the frequency domain represent the frequencies that are most prevalent in the time domain signal.

#### Short-Time Fourier transform (STFT)

In practice, sound is often analyzed using a **Short-Time Fourier Transform (STFT)**, which divides the signal into shorter segments and applies the Fourier Transform to each one.

<img src="./Lecture8 Audio Understanding.assets/image-20240409134236929.png" alt="image-20240409134236929" style="zoom:80%;" />

- $H$: hop length
- $N$: window length

### Spectrogram (声谱图)

The magnitude $|X[m,k]|$​​ of the STFT is called a **spectrogram**.

We can plot a spectrogram like an image.

<img src="./Lecture8 Audio Understanding.assets/image-20240409134423879.png" alt="image-20240409134423879" style="zoom:67%;" />

<img src="./Lecture8 Audio Understanding.assets/image-20240409134453745.png" alt="image-20240409134453745" style="zoom:67%;" />

#### Time vs frequency resolution  tradeoff

<img src="./Lecture8 Audio Understanding.assets/image-20240409134914648.png" alt="image-20240409134914648" style="zoom:67%;" />

- (b) small $N$: poor frequency  resolution, frequencies  localized in time
- (c) large $N$: good frequency  resolution, frequencies  smeared in time
- hop size $H$: smaller hop sizes  give better time resolution,  but more computation time

#### Linear-frequency Spectrogram

<img src="./Lecture8 Audio Understanding.assets/image-20240409135058303.png" alt="image-20240409135058303" style="zoom:67%;" />



#### Log-Frequency Spectrogram

<img src="./Lecture8 Audio Understanding.assets/image-20240409135111403.png" alt="image-20240409135111403" style="zoom:67%;" />

#### Mel-Spectrogram

<img src="./Lecture8 Audio Understanding.assets/image-20240409135133080.png" alt="image-20240409135133080" style="zoom:67%;" />

### Audio signal processing tool (Python)

#### Librosa

<img src="./Lecture8 Audio Understanding.assets/image-20240409135207880.png" alt="image-20240409135207880" style="zoom:50%;" />

<img src="./Lecture8 Audio Understanding.assets/image-20240409135216539.png" alt="image-20240409135216539" style="zoom:80%;" />



#### Audio Datasets Soun[D]ata

[Supported Datasets and Annotations — mirdata 0.3.8 documentation](https://mirdata.readthedocs.io/en/stable/source/quick_reference.html)

[Supported Datasets and Annotations — soundata 0.1.3 documentation](https://soundata.readthedocs.io/en/latest/source/quick_reference.html)

## 2. Audio Classification

<img src="./Lecture8 Audio Understanding.assets/image-20240409135405404.png" alt="image-20240409135405404" style="zoom:67%;" />



### Application 

- **Instrument Identification**
    - Input: recordings of solo instruments, Output: instrument label
    - Input: recordings of songs, Output: labels of active instruments over time
- **Music vs. Speech**
    - Input: audio recording, Output: music or speech label
- **Chord recognition**
    - Input: music recording, Output: chord labels over time
- **Drum Transcription**
    - Input: music recording, Output: active drum instruments over time
- **Music Captioning**
    - Input: music recording, Output: description of the given music. (Ref: “LP-MusicCaps”)
- **Sound tagging**
    - Input: recordings of everyday sounds, Output: sound label (e.g. “dog”)
- **Sound event detection**
    - Input: audio recording, Output: sound label and timestamp (e.g. “car, 1.24s, 2.03s”)
- **Sound event detection and localization**
    - Input: multi-channel audio recording, Output: sound label, timestamp, azimuth and  elevation (e.g. “car, 1.24s, 2.03s, 30°, 15°”)
- **Acoustic scene classification**
    - Input: audio recording, Output: scene label (“e.g. shopping mall”)

### Tasks

#### Classification

<img src="./Lecture8 Audio Understanding.assets/image-20240409135656809.png" alt="image-20240409135656809" style="zoom:67%;" />

#### Classification vs. Detection

<img src="./Lecture8 Audio Understanding.assets/image-20240409135721799.png" alt="image-20240409135721799" style="zoom:50%;" />

#### Convolutional Neural Networks in audio

> [[1607.02444\] Explaining Deep Convolutional Neural Networks on Music Classification (arxiv.org)](https://arxiv.org/abs/1607.02444)

<img src="./Lecture8 Audio Understanding.assets/image-20240409135808375.png" alt="image-20240409135808375" style="zoom: 50%;" />

- Idea
    - learn hierarchies of patterns (from edges to concepts) in an effective manner
    - Texture of spectrogram is closely related to timbre of sound and music
- CNNs expect fixed shape input, what do we do?
    - Force fixed size input spectrograms / signals (e.g. padding / truncating)
    - Split the signal into small windows of fixed size, e.g. 1s (add padding to the last one)

## 3. Data Augmentation

### Dataset Problem

What happens when you don’t have a training dataset for the data you want to apply a model to?

<img src="./Lecture8 Audio Understanding.assets/image-20240409135947562.png" alt="image-20240409135947562" style="zoom:67%;" />

- Training datasets aren’t always representative of what you want to model
- **Available training dataset**: Medley-Solos-DB (which contains standard, good quality music recording)
- **Example application**: Classify the instrument in recordings from Freedsound (which is in a noisy and messy environment)
- **One solution**: Add examples that look more like your test data using Data Augmentation

<img src="./Lecture8 Audio Understanding.assets/image-20240409140158168.png" alt="image-20240409140158168" style="zoom:67%;" />

### Introduction

<img src="./Lecture8 Audio Understanding.assets/image-20240409140258512.png" alt="image-20240409140258512" style="zoom:67%;" />

- **Data Augmentation** is the process of increasing the size of an existing dataset by adding modified examples of the original data.

<img src="./Lecture8 Audio Understanding.assets/image-20240409140318001.png" alt="image-20240409140318001" style="zoom:67%;" />

- The transformations should be **valid** for the task!
- If you change the input, you (sometimes) have to **change the label** to match.
    - Augmentations that **don’t change the label** are called **label-preserving** augmentation
        - E.g. adding a small amount of background noise to a solo  instrument recording doesn’t change the instrument label
    - Other augmentations **deform** the labels
        - E.g. time stretching a song changes the positions of the beats.

### Musical data augmentation

#### Pitch shifting

<img src="./Lecture8 Audio Understanding.assets/image-20240409140604451.png" alt="image-20240409140604451" style="zoom: 50%;" />

- Change the pitch of the audio without changing the speed
    - Label preserving for instrument identification
        - Unless the transformation is extreme
    - Deforms labels for pitch-related data
        - e.g. For chords, the labels  should be shifted along with the  audio!

#### Time stretching

<img src="./Lecture8 Audio Understanding.assets/image-20240409140901500.png" alt="image-20240409140901500" style="zoom:50%;" />

- Change the speed without changing the pitch
    - Label preserving for instrument  identification*
    - Duration changes - for a time stretching factor of $TS$:
        - $TS>1$: speed up, $TS<1$: slow down
        - $d_{new} = \frac{d_{old}}{TS}$
- Deforms labels for time-related labels
    - e.g. For chords, the labels should be shifted along with the audio!

#### Resampling (重采样)

- Change the pitch and speed at the same time
    - conceptually equivalent to playing a record at  a faster or slower speed
- Resampling by a factor of $f$:
    - $f>1$: speed up and increase pitch
    - $f<1$: slow down and decrease pitch
    - duration changes to $\frac{d_{old}}{f}$
    - shift pitch by $12 log_2(f)$
- Labels need to be deformed in both time and  pitch

#### Reverb (混响)

- Add reverb to a recording
- E.g. original recording might be in a  studio, test set might be in a concert  hall.
- Does not affect the labels

#### Relative volume adjustment

- If you have individual recordings of different sources (e.g. solo  vocals + accompaniment) can create new mixes with different relative volumes
- Doesn’t affect labels, unless a source becomes masked by another source

#### Dynamic range compression

<img src="./Lecture8 Audio Understanding.assets/image-20240409141252368.png" alt="image-20240409141252368" style="zoom:67%;" />

- Non-linear volume adjustment
    - Reduces loud sounds, making the overall volume sound louder
    - Does not affect labels

### Some differences

- Recording Conditions
- Recording Quality
- Instrument Characteristics
- Level of musicians
- Musical Style/Genre
- Instrumentation