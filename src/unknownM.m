%% demo_coherent_sources_beamforming.m
% Shows the effect of coherent sources on conventional delay-and-sum beamforming
% Compare: noncoherent vs coherent sources with the same DOAs and SNR

clear; clc; close all;

%% Parameters
fc = 2.4e9;
c  = 3e8;
lambda = c/fc;

M = 16;                 % sensors
d = lambda/2;           % spacing
N = 1000;               % snapshots
SNR_dB = 10;            % SNR

thetas_deg = [-10, 15];  % two close sources (try 10 and 15; or 10 and 30)
K = numel(thetas_deg);

scan_grid = -90:0.1:90;

rng(4);

%% Build steering matrix A
k = 2*pi/lambda;
m = (0:M-1).';
A = zeros(M, K);
for i = 1:K
    th = deg2rad(thetas_deg(i));
    A(:,i) = exp(-1j*k*d*m*sin(th));
end

%% Generate signals: noncoherent vs coherent
% Noncoherent: independent signals
S_non = (randn(K,N) + 1j*randn(K,N))/sqrt(2);

% Coherent: same waveform, scaled copies (multipath-like)
s0 = (randn(1,N) + 1j*randn(1,N))/sqrt(2);
gains = [1.0; 0.8];      % relative path gains
S_coh = gains * s0;

%% Function to add noise at desired SNR
add_noise = @(Xclean) addNoiseSNR(Xclean, SNR_dB);

%% Construct received data
X_non = add_noise(A*S_non);
X_coh = add_noise(A*S_coh);

%% Compute sample covariance matrices
Rx_non = (1/N) * (X_non * X_non');
Rx_coh = (1/N) * (X_coh * X_coh');

%% Conventional beamforming spectra
P_non = das_beamforming_spectrum(Rx_non, d, lambda, scan_grid);
P_coh = das_beamforming_spectrum(Rx_coh, d, lambda, scan_grid);

% Normalize and convert to dB
P_non_dB = 10*log10(P_non / max(P_non));
P_coh_dB = 10*log10(P_coh / max(P_coh));

%% Plot
figure;
plot(scan_grid, P_non_dB, 'LineWidth', 1.3); hold on; grid on;
plot(scan_grid, P_coh_dB, 'LineWidth', 1.3);
xlabel('Angle (deg)'); ylabel('BF Spectrum (dB, normalized)');
title(sprintf('Conventional Beamforming: Noncoherent vs Coherent (M=%d, N=%d, SNR=%g dB)', M, N, SNR_dB));
legend('Noncoherent sources', 'Coherent sources', 'Location', 'best');

% Mark true DOAs
yl = ylim;
for t = thetas_deg
    xline(t, '--', 'LineWidth', 1.0);
end
ylim(yl);

%% Optional: show eigenvalue behavior (to contrast with MUSIC failure)
ev_non = sort(real(eig(Rx_non)), 'descend');
ev_coh = sort(real(eig(Rx_coh)), 'descend');

figure;
stem(1:M, 10*log10(ev_non/max(ev_non)), 'LineWidth', 1.2); hold on; grid on;
stem(1:M, 10*log10(ev_coh/max(ev_coh)), 'LineWidth', 1.2);
xlabel('Eigenvalue index'); ylabel('Eigenvalue (dB, normalized)');
title('Eigenvalues of Rx: Noncoherent vs Coherent (coherence reduces effective rank)');
legend('Noncoherent', 'Coherent', 'Location', 'best');

%% ================== Helper functions ==================

function P = das_beamforming_spectrum(Rx, d, lambda, scan_grid)
% Delay-and-sum beamformer: P(theta) = w^H Rx w, w = a(theta)/M
    M = size(Rx,1);
    k = 2*pi/lambda;
    m = (0:M-1).';
    P = zeros(size(scan_grid));

    for ii = 1:numel(scan_grid)
        th = deg2rad(scan_grid(ii));
        a = exp(-1j*k*d*m*sin(th));
        w = a / M;
        P(ii) = real(w' * Rx * w);
    end
end

function X = addNoiseSNR(Xclean, SNR_dB)
% Adds complex white Gaussian noise to achieve target SNR at the sensor outputs
    signal_power = mean(abs(Xclean(:)).^2);
    noise_power  = signal_power / (10^(SNR_dB/10));
    [M,N] = size(Xclean);
    Noise = sqrt(noise_power/2) * (randn(M,N) + 1j*randn(M,N));
    X = Xclean + Noise;
end
