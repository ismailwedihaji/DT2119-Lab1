# DT2119, Lab 1 Feature Extraction

# Function given by the exercise ----------------------------------
import numpy as np
from scipy.signal import lfilter, hamming
from scipy.fftpack import fft
from lab1_tools import trfbank, lifter
from scipy.fftpack.realtransforms import dct


def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lifter(ceps, liftercoeff)



def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """

    num_frames = 1 + (len(samples) - winlen) // winshift
    frames = np.zeros((num_frames, winlen))
    for i in range(num_frames):
        start = i * winshift
        end = start + winlen
        frames[i, :] = samples[start:end]
    return frames
    
def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    num_frames = input.shape[0]     # How many frames
    output = []                     # List to collect processed frames

    for i in range(num_frames):
            frame = input[i]                             # Get one frame
            filtered_frame = lfilter([1, -p], [1], frame)  # Apply filter
            output.append(filtered_frame)                # Store result

    return np.array(output)  # Convert list to NumPy array and return
def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    num_frames, frame_length = input.shape

   # hamming_window = hamming(frame_length, sym=False)
    # Step 1: Manually create Hamming window of size M
    hamming_window = np.array([
        0.54 - 0.46 * np.cos(2 * np.pi * n / (frame_length - 1))
        for n in range(frame_length)
    ])

    # Step 2: Multiply each frame with the Hamming window
    output = np.zeros_like(input)
    for i in range(num_frames):
        output[i] = input[i] * hamming_window  # element-wise multiplication

    return output

def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
     # Step 1: Apply FFT to each frame along axis 1 (per row)
    fft_result = fft(input, n=nfft, axis=1)

    # Step 2: Compute the power (square of the magnitude)
    power_spec = np.abs(fft_result) ** 2

    return power_spec

def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    # Step 1: Get Mel filterbank (shape: [nmelfilters x nfft])
    filterbank = trfbank(samplingrate, input.shape[1])

    # Step 2: Apply filterbank to each power spectrum frame (matrix multiply)
    mel_energies = np.dot(input, filterbank.T)  # [N x nmelfilters]

    # Step 3: Take log (add small number to avoid log(0))
    log_mel = np.log(mel_energies + 1e-10)

    return log_mel

def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """

    c2 = dct(input, type=2, axis=1)

    # Keep only the first nceps coefficients.
    return c2[:, :nceps]
def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """

    N, M = len(x), len(y)

    # Step 1: Local distances (N x M)
    LD = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            LD[i, j] = dist(x[i], y[j])

    # Step 2: Accumulated distance matrix (N x M)
    AD = np.zeros((N, M))
    AD[0, 0] = LD[0, 0]

    for i in range(1, N):
        AD[i, 0] = LD[i, 0] + AD[i - 1, 0]
    for j in range(1, M):
        AD[0, j] = LD[0, j] + AD[0, j - 1]

    for i in range(1, N):
        for j in range(1, M):
            AD[i, j] = LD[i, j] + min(AD[i - 1, j], AD[i, j - 1], AD[i - 1, j - 1])

    # Step 3: Global distance
    d = AD[N - 1, M - 1] / (N + M)

    return d, LD, AD