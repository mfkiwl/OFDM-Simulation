% Channel estimation using LMS

close all
clear all
clc
%---------------- SIMULATION PARAMETERS ------------------------------------
M = 64;
QAM = log2(M);
SNR_dB = 20; % SNR PER BIT (dB)
NUM_FRAMES = 1*(10^2); % SIMULATION RUNS
FFT_LEN = 1024; % LENGTH OF THE FFT/IFFT
CHAN_LEN = 10; % NUMBER OF CHANNEL TAPS
FADE_VAR_1D = 0.5; % 1D VARIANCE OF THE FADE CHANNEL
PREAMBLE_LEN = 512; % LENGTH OF THE PREAMBLE
CP_LEN = CHAN_LEN-1; % LENGTH OF THE CYCLIC PREFIX
NUM_BIT = QAM*FFT_LEN; % NUMBER OF DATA BITS (OVERALL RATE IS 2)

% SNR PARAMETERS
SNR = 10^(0.1*SNR_dB); % LINEAR SCALE
NOISE_VAR_1D = 0.5*2*2*FADE_VAR_1D*CHAN_LEN/(2*FFT_LEN*SNR); % 1D VARIANCE OF AWGN
%---------------- PREAMBLE GENERATION ------------------------------------
PREAMBLE_A = randi([0 1], QAM * PREAMBLE_LEN, 1);
PREAMBLE_QAM = [];
for i = 1:QAM:length(PREAMBLE_A)
    % Chia chuỗi dữ liệu thành các phần có độ dài QAM
    part = PREAMBLE_A(i:(i + QAM - 1));
    
    % Mã hóa QAM cho từng phần
    s = qammod(part, M, 'InputType', 'bit');
    
    % Gộp các phần đã mã hóa vào chuỗi dữ liệu QAM
    PREAMBLE_QAM = [PREAMBLE_QAM s];
end

% AVERAGE POEWR OF PREAMBLE PART MUST BE EQUAL TO DATA PART
PREAMBLE_QAM = sqrt(PREAMBLE_LEN/FFT_LEN)*PREAMBLE_QAM; 
PREAMBLE_QAM_IFFT = ifft(PREAMBLE_QAM);
%--------------------------------------------------------------------------
C_MSE = 0; % SQUARED ERROR OF CIR AFTER SDM
C_BER = 0; % BIT ERRORS IN EACH FRAME
tic()
%-------------------------------
% estimate of the autocorrelation of the input
Rvv0 = PREAMBLE_QAM_IFFT*PREAMBLE_QAM_IFFT'/(PREAMBLE_LEN);
MAX_STEP_SIZE = 2/(CHAN_LEN*Rvv0); % MAXIMUM STEP SIZE
STEP_SIZE = 0.125*MAX_STEP_SIZE;

%---------------- TRANSMITTER ------------------------------------
for FRAME_CNT = 1:NUM_FRAMES
% SOURCE
A = randi([0 1],NUM_BIT,1); 

% QAM mapping 
MOD_SIG = [];
for i= 1:QAM:length(A)
      s = qammod(A(i:(i+QAM-1)), M,'InputType','bit');
        MOD_SIG = [MOD_SIG s];
end

% IFFT OPERATION
T_QAM_SIG = ifft(MOD_SIG); 

% INSERTING CYCLIC PREFIX AND PREAMBLE
T_TRANS_SIG = [PREAMBLE_QAM_IFFT T_QAM_SIG(end-CP_LEN+1:end) T_QAM_SIG]; 
%---------------- CHANNEL ------------------------------------
% RAYLEIGH FADING CHANNEL
FADE_CHAN = sqrt(FADE_VAR_1D)*randn(1,CHAN_LEN) + 1i*sqrt(FADE_VAR_1D)*randn(1,CHAN_LEN);     

% AWG
AWGN = sqrt(NOISE_VAR_1D)*randn(1,FFT_LEN + CP_LEN + PREAMBLE_LEN + CHAN_LEN - 1) ...
    + 1i*sqrt(NOISE_VAR_1D)*randn(1,FFT_LEN + CP_LEN + PREAMBLE_LEN + CHAN_LEN - 1); 

% CHANNEL OUTPUT
CHAN_OP = conv(T_TRANS_SIG,FADE_CHAN) + AWGN; % Chan_Op stands for channel output
%---------------- RECEIVER ------------------------------------
% CHANNEL ESTIMATION USING LMS
EST_FADE_CHAN = zeros(1,CHAN_LEN); % INITIALIZATION

for i1 = 1:PREAMBLE_LEN-CHAN_LEN+1
INPUT = fliplr(PREAMBLE_QAM_IFFT(i1:i1+CHAN_LEN-1));
ERROR = CHAN_OP(i1+CHAN_LEN-1) - EST_FADE_CHAN*INPUT.';
EST_FADE_CHAN = EST_FADE_CHAN+STEP_SIZE*ERROR*conj(INPUT);
end 

% COMPUTING MEAN SQUARED ERROR OF THE ESTIMATED CHANNEL IMPULSE RESPONSE
ERROR = EST_FADE_CHAN - FADE_CHAN;
C_MSE = C_MSE + ERROR*ERROR';

%-----------------------------------------------------------------------
EST_FREQ_RESP = fft(EST_FADE_CHAN,FFT_LEN);
% discarding preamble
CHAN_OP(1:PREAMBLE_LEN) = [];
% discarding cyclic prefix and transient samples
CHAN_OP(1:CP_LEN) = [];
T_REC_SIG_NO_CP = CHAN_OP(1:FFT_LEN);
% PERFORMING THE FFT
F_REC_SIG_NO_CP = fft(T_REC_SIG_NO_CP);
% ML DETECTION
% Ký hiệu 64-QAM cơ bản
QAM_SYM = [-7-7i, -7-5i, -7-1i, -7-3i, -7+7i, -7+5i, -7+1i, -7+3i, ...
              -5-7i, -5-5i, -5-1i, -5-3i, -5+7i, -5+5i, -5+1i, -5+3i, ...
              -1-7i, -1-5i, -1-1i, -1-3i, -1+7i, -1+5i, -1+1i, -1+3i, ...
              -3-7i, -3-5i, -3-1i, -3-3i, -3+7i, -3+5i, -3+1i, -3+3i, ...
               7-7i,  7-5i,  7-1i,  7-3i,  7+7i,  7+5i,  7+1i,  7+3i, ...
               5-7i,  5-5i,  5-1i,  5-3i,  5+7i,  5+5i,  5+1i,  5+3i, ...
               1-7i,  1-5i,  1-1i,  1-3i,  1+7i,  1+5i,  1+1i,  1+3i, ...
               3-7i,  3-5i,  3-1i,  3-3i,  3+7i,  3+5i,  3+1i,  3+3i];

% Khởi tạo ma trận khoảng cách
DIST_QAM = zeros(64, FFT_LEN);

% Tính toán khoảng cách giữa tín hiệu nhận được và các ký hiệu 64-QAM cơ bản
for i = 1:64
    DIST_QAM(i, :) = abs(F_REC_SIG_NO_CP - EST_FREQ_RESP.*QAM_SYM(i)).^2;
end

% Phát hiện ML: Tìm ký hiệu 64-QAM gần nhất
[~, INDICES] = min(DIST_QAM, [], 1);

% MAPPING INDICES TO QAM SYMBOLS
DEC_QAM_MAP_SYM = QAM_SYM(INDICES);
% DEMAPPING QAM SYMBOLS TO BITS
DEC_A = [];
for i=1:length(DEC_QAM_MAP_SYM)
    dataR = qamdemod(DEC_QAM_MAP_SYM(i), M, 'OutputType', 'bit');
    DEC_A=[DEC_A dataR];
end
DEC_A = reshape(DEC_A,[],1);
% CALCULATING BIT ERRORS IN EACH FRAME
C_BER = C_BER + nnz(A-DEC_A);
end
toc()
% bit error rate
BER = C_BER/(NUM_BIT*NUM_FRAMES)

% MEAN SQUARE ERROR OF THE CHANNEL ESTIMATION
MSE = C_MSE/(CHAN_LEN*NUM_FRAMES)