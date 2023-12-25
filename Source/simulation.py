import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_symbols = 1000
num_carriers = 64
snr_dB = 20

# Generate random binary data
data_bits = np.random.randint(0, 2, num_symbols)

# Modulation: 64 QAM
modulation_order = 64
modulated_symbols = np.random.choice([-3-3j, -3-1j, -3+1j, -3+3j, -1-3j, -1-1j, -1+1j, -1+3j, 1-3j, 1-1j, 1+1j, 1+3j, 3-3j, 3-1j, 3+1j, 3+3j], num_symbols)

# OFDM modulation
ofdm_symbols = np.zeros((num_symbols, num_carriers), dtype=complex)
ofdm_symbols[:, :modulation_order] = modulated_symbols.reshape(-1, modulation_order)

# IFFT
ofdm_time_domain = np.fft.ifft(ofdm_symbols, axis=1)

# Transmit signal through AWGN channel
noise_power = 10 ** (-snr_dB / 10)
noise = np.sqrt(noise_power / 2) * (np.random.randn(*ofdm_time_domain.shape) + 1j * np.random.randn(*ofdm_time_domain.shape))
received_signal = ofdm_time_domain + noise

# FFT at the receiver
received_symbols = np.fft.fft(received_signal, axis=1)

# Demodulation: 64 QAM
demodulated_symbols = received_symbols[:, :modulation_order].flatten()

# BER calculation
error_bits = np.sum(data_bits != np.real(demodulated_symbols) > 0)
ber = error_bits / (num_symbols * modulation_order)

print(f"Bit Error Rate (BER): {ber}")

# Plot original and received symbols
plt.scatter(np.real(modulated_symbols), np.imag(modulated_symbols), label='Original Symbols')
plt.scatter(np.real(demodulated_symbols), np.imag(demodulated_symbols), label='Received Symbols', marker='x')
plt.title('64 QAM Modulation and Demodulation')
plt.xlabel('I')
plt.ylabel('Q')
plt.legend()
plt.show()