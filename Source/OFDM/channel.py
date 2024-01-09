import numpy as np
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from scipy.fftpack import fft
# from .helper import PS_FIXED


def group_bits(bits, bits_per_group):
    bits_groups = []
    x = 0
    num_groups = (len(bits)//bits_per_group)+1
    for i in range(num_groups):
        bits_groups.append(bits[x:x+bits_per_group])
        x = x+bits_per_group
    bits_groups[-1] = np.zeros(bits_per_group, dtype = int)
    return bits_groups


def group_bits_zero_pad(bits, bits_per_group):
    num_groups = (len(bits) + bits_per_group - 1) // bits_per_group
    # Pad the bits list with zeros if needed
    bits_zero_pad = bits
    bits_zero_pad.resize(num_groups*bits_per_group, refcheck=False)
    # Use a list comprehension to create the groups
    bits_groups = [bits_zero_pad[i:i + bits_per_group] for i in range(0, len(bits), bits_per_group)]
    return bits_groups


def S2P(bits_serial : np.ndarray, length : int, mu : int):
    return bits_serial.reshape(length, mu)


def P2S(bits_parallel : np.ndarray):
    return bits_parallel.reshape((-1,))


def Mapping(bits, mapping_table):
    return np.array([mapping_table[tuple(b)] for b in bits])


def Demapping(QAM, demapping_table):
    # array of possible constellation points
    constellation = np.array([x for x in demapping_table.keys()])
    # calculate distance of each RX point to each possible point
    dists = abs(QAM.reshape((-1,1)) - constellation.reshape((1,-1)))
    # for each element in QAM, choose the index in constellation
    # that belongs to the nearest constellation point
    const_index = dists.argmin(axis=1)
    # get back the real constellation point
    hardDecision = constellation[const_index]
    # transform the constellation point into the bit groups
    return np.vstack([demapping_table[C] for C in hardDecision]), hardDecision


def OFDM_symbol(QAM_payload, K, dataCarriers, pilotCarriers, pilotValue):
    symbol = np.zeros(K, dtype=complex) # the overall K subcarriers
    symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers
    symbol[dataCarriers] = QAM_payload  # allocate the pilot subcarriers
    return symbol


def FFT(OFDM_RX, n=None):
    return np.fft.fft(OFDM_RX, n)


def IFFT(OFDM_data):
    return np.fft.ifft(OFDM_data)


def addCP(symbol, CP):
    cp = symbol[-CP:]               # take the last CP samples ...
    return np.hstack([cp, symbol])  # append them to the beginning


def removeCP(symbol, CP, K):
    return symbol[CP:(CP+K)]


def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest


def get_payload(equalized, dataCarriers):
    return equalized[dataCarriers]


def channel(signal, channelResponse, snrdb):
    convolved = np.convolve(signal, channelResponse)
    # convolved = fftconvolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-snrdb/10) # calculate noise power based on signal power and SNR
    # print ("RX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))
    # Generate complex noise with given variance
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    return convolved + noise


def channelEstimate(OFDM_demod, allCarriers, pilotCarriers, pilotValue):
    '''
    Perform interpolation between the pilot carriers to get an estimate
    of the channel in the data carriers. Here, we interpolate absolute value and phase
    separately
    '''
    pilots = OFDM_demod[pilotCarriers]      # extract the pilot values from the RX signal
    Hest_at_pilots = pilots / pilotValue    # divide by the transmitted pilot values
    Hest_abs = interp1d(pilotCarriers, abs(Hest_at_pilots), kind='linear')(allCarriers)
    Hest_phase = interp1d(pilotCarriers, np.angle(Hest_at_pilots), kind='linear')(allCarriers)
    Hest = Hest_abs * np.exp(1j*Hest_phase)
    return Hest, Hest_at_pilots