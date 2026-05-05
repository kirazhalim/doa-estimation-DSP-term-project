% last file: FBSS simulation.
%% DOA Estimation Project: BF vs MUSIC + Performance Analysis + Spatial Smoothing Extension
clear; clc; close all;

%% ===================== 1. BASIC PARAMETERS ======================
c        = 3e8;          % Speed of light (m/s)
fc       = 2.4e9;        % Carrier frequency (Hz)
lambda   = c / fc;       % Wavelength (m)
d        = lambda / 2;   % Antenna spacing (half wavelength)
M_default      = 16;     % Increased slightly to allow for subarrays
K              = 2;      % Number of sources
doas_deg_true  = [-10 5];       % True DOAs (degrees)
N_snap_default = 1000;           % Number of snapshots
SNR_dB_default = 10;             % SNR for single example
theta_scan = -90:0.5:90;         % Scanning angles

%% ===================== 2. ORIGINAL SCENARIO (IDEAL) ======================
% This block generates X, computes Rx, BF and MUSIC, and shows spectra.
fprintf('--- Running Original Scenario (Ideal Uncorrelated Sources) ---\n');

% Generate signals (Uncorrelated) and array data
S = generate_signals(K, N_snap_default);
A = steering_matrix_ula(doas_deg_true, M_default, d, lambda);
[X, noise_power] = generate_data(A, S, SNR_dB_default);
fprintf('Single example: noise power = %.4e\n', noise_power);

% Sample covariance
Rx = (X * X') / N_snap_default;

% BF and MUSIC spectra
P_BF  = conventional_bf(Rx, M_default, d, lambda, theta_scan);
P_MU  = music_spectrum(Rx, K, d, lambda, theta_scan);

% Normalize and convert to dB
P_BF  = P_BF  / max(P_BF);
P_MU  = P_MU  / max(P_MU);
P_BF_dB = 10*log10(P_BF + eps);
P_MU_dB = 10*log10(P_MU + eps);

% DOA estimation from spectra
doa_bf_est  = estimate_doa_from_spectrum(theta_scan, P_BF_dB, K);
doa_mu_est  = estimate_doa_from_spectrum(theta_scan, P_MU_dB, K);

fprintf('True DOAs:       '); fprintf('%6.2f ', doas_deg_true); fprintf('\n');
fprintf('BF estimated:    '); fprintf('%6.2f ', doa_bf_est);    fprintf('\n');
fprintf('MUSIC estimated: '); fprintf('%6.2f ', doa_mu_est);    fprintf('\n');


%% ===================== 3. EXTENSION: COHERENT SIGNALS & SPATIAL SMOOTHING ======================
% In this section, we simulate a "Multipath" scenario where sources are coherent.
% Standard MUSIC will fail. We then use Spatial Smoothing to fix it.

fprintf('\n--- Running Extension (Coherent Sources & Spatial Smoothing) ---\n');

% 1. Generate Coherent Signals
% Source 2 is now a copy of Source 1 (perfectly correlated/coherent)
% We reuse the function generate_signals for S1, then manually create S2.
s1 = (randn(1, N_snap_default) + 1j*randn(1, N_snap_default))/sqrt(2);
complex_fading = 0.9 * exp(1j * pi/4); % Attenuation and phase shift (Multipath)
s2 = complex_fading * s1;              % Source 2 is a copy of Source 1
S_coh = [s1; s2];

% Generate Data for Coherent Signals
% Note: We use the same 'A' matrix and SNR as the original scenario
[X_coh, ~] = generate_data(A, S_coh, SNR_dB_default);

% 2. Run Standard MUSIC (Expected to FAIL)
Rx_coh = (X_coh * X_coh') / N_snap_default;
P_MU_fail = music_spectrum(Rx_coh, K, d, lambda, theta_scan);
P_MU_fail_dB = 10*log10(P_MU_fail / max(P_MU_fail) + eps);

% 3. Run Spatial Smoothing MUSIC (The Fix)
M_sub = 7; % Subarray size. Note: M_sub must be > K. 
           % Resolution will degrade slightly (aperture M -> M_sub)


           
Rx_ss = forward_backward_spatial_smoothing(X_coh, M_sub);

% Run MUSIC on the smoothed matrix
% Note: The array size for MUSIC is now M_sub, not M_default.
P_MU_ss = music_spectrum(Rx_ss, K, d, lambda, theta_scan);
P_MU_ss_dB = 10*log10(P_MU_ss / max(P_MU_ss) + eps);

% 4. Plot The Comparison
figure('Position', [100, 100, 900, 600]);

% Subplot 1: Failure of Standard MUSIC
subplot(2,1,1);
plot(theta_scan, P_MU_fail_dB, 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1.2); % Matches MUSIC color above
hold on;
yL = ylim;
for k = 1:K
    plot([doas_deg_true(k) doas_deg_true(k)], yL, 'k--');
end
grid on;
title('Coherent Signals with Standard MUSIC');
xlabel('\theta (deg)'); 
ylabel('Normalized spectrum (dB)');
legend('Standard MUSIC', 'True DOAs');

% Subplot 2: Success of Spatial Smoothing
subplot(2,1,2);
plot(theta_scan, P_MU_ss_dB, 'Color', [0 0.4470 0.7410], 'LineWidth', 1.2); % Matches BF color (Blue) for contrast
hold on;
yL = ylim;
for k = 1:K
    plot([doas_deg_true(k) doas_deg_true(k)], yL, 'k--');
end
grid on;
title(['Forward-Backward Spatial Smoothing (Subarray Size: ' num2str(M_sub) ')']);
xlabel('\theta (deg)'); 
ylabel('Normalized spectrum (dB)');
legend('SS-MUSIC', 'True DOAs');


%% ===================== 4. LOCAL FUNCTIONS ======================
% (Must be at the end of the script)

function Rx_ss = forward_backward_spatial_smoothing(X, M_sub)
% Performs Forward-Backward Spatial Smoothing to decorrelate coherent signals.
% Inputs:
%   X: Received Signal Matrix (M x N_snap)
%   M_sub: Size of the subarrays
% Output:
%   Rx_ss: Smoothed Covariance Matrix (M_sub x M_sub)

    [M, N_snap] = size(X);
    L = M - M_sub + 1; % Number of overlapping subarrays
    
    Rx_fwd = zeros(M_sub, M_sub);
    
    % Forward Smoothing
    for i = 1:L
        X_sub = X(i : i+M_sub-1, :); % Sliding window
        R_sub = (X_sub * X_sub') / N_snap;
        Rx_fwd = Rx_fwd + R_sub;
    end
    Rx_fwd = Rx_fwd / L;
    
    % Backward Smoothing
    % Exchange matrix J flips the matrix anti-diagonally
    J = fliplr(eye(M_sub)); 
    Rx_bwd = J * conj(Rx_fwd) * J;
    
    % Average both to get final smoothed matrix
    Rx_ss = (Rx_fwd + Rx_bwd) / 2;
end

function S = generate_signals(K, N_snap)
% Generate K independent complex baseband signals (CN(0,1)).
    S = (randn(K, N_snap) + 1j*randn(K, N_snap))/sqrt(2); 
end

function A = steering_matrix_ula(doas_deg, M, d, lambda)
% Build steering matrix for ULA.
    K = numel(doas_deg); 
    A = zeros(M, K);
    k = 2*pi / lambda;
    m_vec = (0:M-1).';
    for idx = 1:K
        theta = deg2rad_local(doas_deg(idx));
        A(:, idx) = exp(-1j * k * d * m_vec * sin(theta)); 
    end
end

function [X, noise_power] = generate_data(A, S, SNR_dB)
% Generate array data X = A*S + W with desired SNR.
    [M, ~]     = size(A);
    [~, Nsnap] = size(S);
    X_sig  = A * S;
    sig_p  = mean(abs(X_sig(:)).^2);
    snrLin = 10^(SNR_dB/10);
    noise_power = sig_p / snrLin;
    W = sqrt(noise_power/2) * (randn(M, Nsnap) + 1j*randn(M, Nsnap));
    X = X_sig + W;
end

function P_BF = conventional_bf(Rx, M, d, lambda, theta_scan)
% Delay-and-sum beamformer spatial spectrum from covariance Rx.
    k = 2*pi / lambda;
    P_BF = zeros(size(theta_scan));
    m_vec = (0:M-1).';
    for i = 1:length(theta_scan)
        theta = deg2rad_local(theta_scan(i));
        a = exp(-1j * k * d * m_vec * sin(theta));
        w = a / M;
        P_BF(i) = real(w' * Rx * w);
    end
end

function P_MU = music_spectrum(Rx, K, d, lambda, theta_scan)
% MUSIC pseudo-spectrum from covariance Rx.
    % Note: This function automatically adapts to the size of Rx (M or M_sub)
    [M_curr, ~] = size(Rx); 
    [E, D] = eig(Rx);
    [~, idx] = sort(diag(D), 'descend');
    E = E(:, idx);
    En = E(:, K+1:end);   % noise subspace
    
    k = 2*pi / lambda;
    m_vec = (0:M_curr-1).'; % ADAPTED: Uses size of Rx, not fixed M
    
    P_MU = zeros(size(theta_scan));
    for i = 1:length(theta_scan)
        theta = deg2rad_local(theta_scan(i));
        a = exp(-1j * k * d * m_vec * sin(theta));
        denom = sum(abs(En' * a).^2);
        P_MU(i) = 1 / (denom + eps);
    end
end

function doa_est = estimate_doa_from_spectrum(theta_scan, spectrum_dB, K)
% Estimate DOAs from spatial spectrum by picking top K peaks.
    doa_est = [];
    if any(isnan(spectrum_dB)) || all(~isfinite(spectrum_dB))
        return;
    end
    if exist('findpeaks', 'file')
        [~, locs] = findpeaks(spectrum_dB, theta_scan, ...
                              'SortStr', 'descend', 'NPeaks', K);
        doa_est = sort(locs(:).');
    else
        [~, idx_sorted] = sort(spectrum_dB, 'descend');
        idx_top = idx_sorted(1:K);
        doa_est = sort(theta_scan(idx_top));
    end
end

function rmse_val = compute_rmse(true_deg, est_deg)
% RMSE between two DOA sets.
    true_deg = sort(true_deg(:).');
    if isempty(est_deg)
        rmse_val = NaN;
        return;
    end
    est_deg  = sort(est_deg(:).');
    L = min(numel(true_deg), numel(est_deg));
    diff_sq = (true_deg(1:L) - est_deg(1:L)).^2;
    rmse_val = sqrt(mean(diff_sq));
end

function y = deg2rad_local(x)
    y = x * pi / 180;
end