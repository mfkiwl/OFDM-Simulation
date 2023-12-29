import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.fftpack
import math as math

def group_bits(bitc):
    bity = []
    x = 0
    for i in range((len(bitc)//payloadBits_per_OFDM)+1):
        bity.append(bitc[x:x+payloadBits_per_OFDM])
        x = x+payloadBits_per_OFDM
    k = i-1
    pp = np.zeros(714, dtype = int)
    bity[-1] = pp
    return bity

def SP(bits):
    return bits.reshape((len(dataCarriers), mu))


def Mapping(bits):
    return np.array([mapping_table[tuple(b)] for b in bits])

def OFDM_symbol(QAM_payload):
    symbol = np.zeros(K, dtype=complex) # the overall K subcarriers
    symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
    symbol[dataCarriers] = QAM_payload  # allocate the pilot subcarriers
    return symbol

def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)


def addCP(OFDM_time):
    cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # add them to the beginning
 
def channel(signal):
    convolved = np.convolve(signal,channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb/10) # calculate noise power based on signal power and SNR
    print ("RX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))     
    # Generate complex noise with given variance
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    return convolved + noise

def removeCP(signal):
    return signal[CP:(CP+K)]

def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)

def channelEstimate(OFDM_demod):
    pilots = OFDM_demod[pilotCarriers]  # extract the pilot values from the RX signal
    Hest_at_pilots = pilots / pilotValue # divide by the transmitted pilot values
    
    # Perform interpolation between the pilot carriers to get an estimate
    # of the channel in the data carriers. Here, we interpolate absolute value and phase 
    # separately
    Hest_abs = scipy.interpolate.interp1d(pilotCarriers, abs(Hest_at_pilots), kind='linear')(allCarriers)
    Hest_phase = scipy.interpolate.interp1d(pilotCarriers, np.angle(Hest_at_pilots), kind='linear')(allCarriers)
    Hest = Hest_abs * np.exp(1j*Hest_phase)
    
    plt.plot(allCarriers, abs(H_exact), label='Correct Channel')
    plt.stem(pilotCarriers, abs(Hest_at_pilots), label='Pilot estimates')
    plt.plot(allCarriers, abs(Hest), label='Estimated channel via interpolation')
    plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
    plt.ylim(0,2)
    
    return Hest
def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest

def get_payload(equalized):
    return equalized[dataCarriers]

def Demapping(QAM):
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

def PS(bits):
    return bits.reshape((-1,))

def channelEstimate_FIXED(OFDM_demod):
    pilots = OFDM_demod[pilotCarriers]  # extract the pilot values from the RX signal
    Hest_at_pilots = pilots / pilotValue # divide by the transmitted pilot values
    
    # Perform interpolation between the pilot carriers to get an estimate
    # of the channel in the data carriers. Here, we interpolate absolute value and phase 
    # separately
    Hest_abs = scipy.interpolate.interp1d(pilotCarriers, abs(Hest_at_pilots), kind='linear')(allCarriers)
    Hest_phase = scipy.interpolate.interp1d(pilotCarriers, np.angle(Hest_at_pilots), kind='linear')(allCarriers)
    Hest = Hest_abs * np.exp(1j*Hest_phase)   
    return Hest

def SNR_return(num_snr):
    def channel_V(signal):
        convolved = np.convolve(signal,channelResponse)
        signal_power = np.mean(abs(convolved**2))
        sigma2 = signal_power * 10**(-SNRdb/10) # calculate noise power based on signal power and SNR
        # Generate complex noise with given variance
        noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
        return convolved + noise
    SNR_Array = np.arange(0,num_snr,0.05)
    bitsnr = bitx[10]
    ber = []
    for i in SNR_Array:
        SNRdb = i
        bits_SP = SP(bitsnr)
        QAM = Mapping(bits_SP)
        OFDM_data = OFDM_symbol(QAM)
        OFDM_time = IDFT(OFDM_data)
        OFDM_withCP = addCP(OFDM_time)
        OFDM_Tx = OFDM_withCP
        OFDM_Rx = channel_V(OFDM_withCP)
        OFDM_RX_noCP = OFDM_Rx[(CP):(CP+K)]
        OFDM_demod = DFT(OFDM_RX_noCP)
        Hest = channelEstimate_FIXED(OFDM_demod)
        equalized_Hest = equalize(OFDM_demod, Hest)
        QAM_est = get_payload(equalized_Hest)
        PS_est, hardDecision = Demapping(QAM_est)
        bits_est = PS_FIXED(PS_est)
        ber.append(np.sum(abs(bitsnr-bits_est))/len(bitsnr))
    return SNR_Array, ber

def bit_plot(bit_x, a ,b):
    sin_time = []
    #import math as math
    # Number of sample points
    N = 400
    # sample spacing
    T = 1 / 0.32e12
    t = np.linspace(0.0, N*T, N)
    fd = 0
    fc = 24e9
    s = np.zeros(len(t), dtype = float)
    bits_SP = SP(bitx[bit_x])
    QAM = Mapping(bits_SP)
    OFDM_data = OFDM_symbol(QAM)
    OFDM_time_V = IDFT(OFDM_data)
    for i in OFDM_time_V:
        sin_time.append(abs(i)*np.sin(((2*np.pi)*(fc+fd)*t) + math.atan(i.imag/i.real)))
        fd += 1e9
    for j in range(len(t)):
        for i in sin_time:
            s[j] += i[j]
    plt.plot(t,s)
    plt.grid(True)
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude(V)")
    plt.title("OFDM in time-domain")
    plt.show()
    #part2
    for i in range(a,b):
        plt.plot(t, sin_time[i])
        plt.grid(True)
    plt.show()
    #part3
    for i in range(len(sin_time)):
        yf = scipy.fftpack.fft(sin_time[i])
        xf = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
        plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.grid(True)
    plt.title("OFDM in frequency-domain")
    plt.ylabel("|H(f)|")
    plt.xlabel("Frequency(Hz)")
    plt.show()


    #look closely in 2 frequncies spiked
    # Number of sample points
    sin_time_x = []
    G = 75 #N
    # sample spacing
    M = 1 / 0.0580e12 #T
    O = np.linspace(0.0, G*M, G) #t
    fd = 0
    fc = 24e9
    s = np.zeros(len(O), dtype = float)
    for i in OFDM_time_V:
        sin_time_x.append(abs(i)*np.sin(((2*np.pi)*(fc+fd)*O) + math.atan(i.imag/i.real)))
        fd += 1e9
    for i in range(len(sin_time[0:2])):
        yf = scipy.fftpack.fft(sin_time[i])
        xf = np.linspace(0.0, 1.0/(2.0*M), int(G/2))
        plt.plot(xf, 2.0/G * np.abs(yf[:G//2]))
    plt.grid(True)
    plt.title("OFDM in frequency-domain")
    plt.ylabel("|H(f)|")
    plt.xlabel("Frequency(Hz)")
    plt.show()

def channel_X(signal):
    convolved = np.convolve(signal,channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb/10) # calculate noise power based on signal power and SNR
    # Generate complex noise with given variance
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    return convolved + noise

def pro_bits(SNR_X, bit):
    ber = 0
    bit_rx1 = []
    SNRdb = SNR_X
    for i in bit:
        bits_SP = SP(i)
        QAM = Mapping(bits_SP)
        OFDM_data = OFDM_symbol(QAM)
        OFDM_time = IDFT(OFDM_data)
        OFDM_withCP = addCP(OFDM_time)
        OFDM_Tx = OFDM_withCP
        OFDM_Rx = channel_X(OFDM_withCP)
        OFDM_RX_noCP = OFDM_Rx[(CP):(CP+K)]
        OFDM_demod = DFT(OFDM_RX_noCP)
        Hest = channelEstimate_FIXED(OFDM_demod)
        equalized_Hest = equalize(OFDM_demod, Hest)
        QAM_est = get_payload(equalized_Hest)
        PS_est, hardDecision = Demapping(QAM_est)
        bits_est = PS_FIXED(PS_est)
        ber += np.sum(abs(i-bits_est))/len(i)
        bit_rx1.append(bits_est)
    print("Total BER is :" + str(ber / ((len(img_rft)//payloadBits_per_OFDM)+1)))
    print("All frames are sent")
    return bit_rx1
