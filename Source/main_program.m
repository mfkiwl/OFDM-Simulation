clc;
clear all;
close all;
%% Tham so OFDM
    Nfft=32; Ng=Nfft/8; Nofdm=Nfft+Ng; Nsym=100;
    Nps=4; Np=Nfft/Nps; % Pilot spacing and number of pilots per OFDM symbol
    Nbps=6; M=2^Nbps; % Number of bits per (modulated) symbol
 
    Es=1; A=sqrt(3/2/(M-1)*Es); % Signal energy and QAM normalization factor
    SNR = 30; sq2=sqrt(2); MSE = zeros(1,2); nose = 0;
for nsym=1:Nsym
    Xp = 2*(randn(1,Np)>0)-1; % Pilot sequence generation
    msgint=randi([1 M-1],(Nfft-Np)*M,1); % bit generation
    Data = A*qammod(msgint,M);
    ip = 0; pilot_loc = [];
for k=1:Nfft
    if mod(k,Nps)==1
        X(k)=Xp(floor(k/Nps)+1); pilot_loc=[pilot_loc k]; ip = ip+1;
    else 
        X(k) = Data(k-ip);
    end
end
    x = ifft(X,Nfft); xt = [x(Nfft-Ng+1:Nfft) x]; % IFFT and add CP
    h = [(randn+j*randn) (randn+j*randn)/2]; % A (2-tap) channel
    H = fft(h,Nfft); ch_length=length(h); % True channel and its length
    H_power_dB = 10*log10(abs(H.*conj(H))); % True channel power in dB
    y_channel = conv(xt,h); % Channel path (convolution)
    yt = awgn(y_channel,SNR,'measured');
    y = yt(Ng+1:Nofdm); Y = fft(y); % Remove CP and FFT
     H_est = LMSE_CE(Y,Xp,pilot_loc,Nfft,Nps,h,SNR);
    method='MMSE'; % MMSE estimation

    H_est_power_dB = 10*log10(abs(H_est.*conj(H_est)));
    h_est = ifft(H_est); h_DFT = h_est(1:ch_length);
    H_DFT = fft(h_DFT,Nfft); % DFT-based channel estimation
    H_DFT_power_dB = 10*log10(abs(H_DFT.*conj(H_DFT)));
    if nsym==1
        subplot(319+2), plot(H_power_dB,'b'); hold on;
        plot(H_est_power_dB,'r:+'); legend('True Channel',method);
        subplot(320+2), plot(H_power_dB,'b'); hold on;
        plot(H_DFT_power_dB,'r:+');
        legend('True Channel',[method 'with DFT']);
    end
MSE= MSE + (H-H_est)*(H-H_est)';
MSE(2) = MSE(2) + (H-H_DFT)*(H-H_DFT)';

Y_eq = Y./H_est; ip = 0;
for k=1:Nfft
    if mod(k,Nps)==1
        ip=ip+1; 
    else 
        Data_extracted(k-ip)=Y_eq(k); 
    end
end
msg_detected = qamdemod(Data_extracted/A,M);
nose = nose + sum(msg_detected~=msgint);
MSEs = MSE/(Nfft*Nsym);
end

