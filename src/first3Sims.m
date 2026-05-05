%% DOA Estimation Report Generation Script
% This script generates 4 specific scenarios ranging from basic to failure.
clear; clc; close all;
% --- GLOBAL PARAMETERS ---
c = 3e8; fc = 2.4e9; lambda = c/fc; d = lambda/2;
M = 16;             % 16 Antennas
N_snap = 1000;      % 1000 Snapshots
theta_scan = -90:0.1:90; % High resolution scan

%% EXPERIMENT 1: BASIC VALIDATION (Single Source)
% Goal: Prove both algorithms work in ideal conditions.
fprintf('Running Exp 1: Single Source...\n');
K = 1; DOAs = [10]; SNR = 10;
[P_BF, P_MU] = run_simulation(M, K, DOAs, N_snap, SNR, d, lambda, theta_scan);
plot_results(theta_scan, P_BF, P_MU, DOAs, 'Simulation 1', K, M);
saveas(gcf, 'Exp1_SingleSource.png');

%% EXPERIMENT 2: NOISE RESILIENCE (Low SNR)
% Goal: Show performance when noise is high (Signal is weaker than noise).
fprintf('Running Exp 3: Low SNR...\n');
K = 2; DOAs = [-10 30]; SNR = 10; % Negative SNR
[P_BF, P_MU] = run_simulation(M, K, DOAs, N_snap, SNR, d, lambda, theta_scan);
plot_results(theta_scan, P_BF, P_MU, DOAs, 'Simulation 2', K, M);
saveas(gcf, 'Exp3_LowSNR.png');

%% EXPERIMENT 3: RESOLUTION TEST (Closely Spaced)
% Goal: Show BF failing to separate sources while MUSIC succeeds.
fprintf('Running Exp 3: Resolution Test...\n');
K = 2; DOAs = [10 15]; SNR = 10; % 5 degrees separation
[P_BF, P_MU] = run_simulation(M, K, DOAs, N_snap, SNR, d, lambda, theta_scan);
plot_results(theta_scan, P_BF, P_MU, DOAs, 'Simulation 3', K, M);
saveas(gcf, 'Exp2_Resolution.png');

%% --- HELPER FUNCTIONS ---
function [P_BF_dB, P_MU_dB] = run_simulation(M, K, doas, N, SNR, d, lambda, scan)
    % 1. Generate Signals
    S = (randn(K, N) + 1j*randn(K, N))/sqrt(2);
    
    % 2. Steering Matrix (True)
    A = zeros(M, K);
    k_wave = 2*pi/lambda;
    for i = 1:K
        A(:,i) = exp(-1j * k_wave * d * (0:M-1)' * sin(deg2rad(doas(i))));
    end
    
    % 3. Generate Data (X)
    X_sig = A * S;
    sig_power = mean(abs(X_sig(:)).^2);
    noise_power = sig_power / (10^(SNR/10));
    Noise = sqrt(noise_power/2) * (randn(M, N) + 1j*randn(M, N));
    X = X_sig + Noise;
    
    % 4. Covariance Matrix
    Rx = (X*X')/N;
    
    % 5. BF Spectrum
    P_BF = zeros(size(scan));
    m_vec = (0:M-1).';
    for i = 1:length(scan)
        a_theta = exp(-1j * k_wave * d * m_vec * sin(deg2rad(scan(i))));
        w = a_theta / M;
        P_BF(i) = real(w' * Rx * w);
    end
    
    % 6. MUSIC Spectrum
    [E, D] = eig(Rx);
    [~, idx] = sort(diag(D), 'descend');
    E = E(:, idx);
    En = E(:, K+1:end); % Noise subspace
    P_MU = zeros(size(scan));
    for i = 1:length(scan)
        a_theta = exp(-1j * k_wave * d * m_vec * sin(deg2rad(scan(i))));
        denom = sum(abs(En' * a_theta).^2);
        P_MU(i) = 1 / (denom + eps);
    end
    
    % Normalize
    P_BF_dB = 10*log10(P_BF / max(P_BF));
    P_MU_dB = 10*log10(P_MU / max(P_MU));
end

function plot_results(scan, BF, MU, true_doas, prefix_str, K, M)
    figure('Position', [100, 100, 800, 400]);
    plot(scan, BF, 'LineWidth', 1.5, 'DisplayName', 'Beamforming'); hold on;
    plot(scan, MU, 'LineWidth', 1.5, 'DisplayName', 'MUSIC');
    
    % Mark True DOAs
    ylim_vals = ylim;
    for k = 1:length(true_doas)
        xline(true_doas(k), '--k', 'LineWidth', 1, 'HandleVisibility', 'off');
    end
    
    grid on; legend show;
    xlabel('Angle (Degrees)'); ylabel('Normalized Spectrum (dB)');
    
    % Create dynamic title string (SNR REMOVED)
    doas_str = num2str(true_doas, '%g '); % Convert DOAs array to string
    title_full = sprintf('%s: K=%d, DOAs=[%s], M=%d', ...
                         prefix_str, K, strtrim(doas_str), M);
    title(title_full);
    
    xlim([min(true_doas)-30, max(true_doas)+30]); % Zoom in on the action
end