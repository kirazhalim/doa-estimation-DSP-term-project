% second file: comparison simulations
%% DOA Estimation: Extended Parameter Analysis
% This script simulates the effects of Antenna Count, SNR, and Snapshots
% as proposed in the methodology[cite: 31, 32].

clear; clc; close all;

% --- GLOBAL CONSTANTS ---
c = 3e8; fc = 2.4e9; lambda = c/fc; d = lambda/2;
theta_scan = -90:0.1:90; % Scan vector

%% SIMULATION 4: EFFECT OF ANTENNA NUMBER (M)
% Objective: Show that low element count fails to resolve sources, while high count succeeds.
fprintf('Running Sim 4: Effect of Antenna Number...\n');

% Common Parameters
K = 2; DOAs = [10 20]; % Sources 10 degrees apart
N_snap = 1000; SNR = 10;

% Case A: FAIL (Too few antennas to resolve 10 deg separation)
M_fail = 4; 
[P_BF_fail, P_MU_fail] = run_simulation(M_fail, K, DOAs, N_snap, SNR, d, lambda, theta_scan);

% Case B: SUCCESS (Many antennas provide sharp resolution)
M_success = 16; 
[P_BF_succ, P_MU_succ] = run_simulation(M_success, K, DOAs, N_snap, SNR, d, lambda, theta_scan);

% Plotting
create_comparison_plot('Effect of Antenna Number (M)', theta_scan, ...
    P_BF_fail, P_MU_fail, M_fail, SNR, N_snap, ...
    P_BF_succ, P_MU_succ, M_success, SNR, N_snap, DOAs);


%% SIMULATION 5: EFFECT OF SNR
% Objective: Show that high noise buries the signal peaks.
fprintf('Running Sim 5: Effect of SNR...\n');

% Common Parameters
K = 2; DOAs = [-10 15]; 
M = 16; N_snap = 1000;

% Case A: FAIL (Signal buried in noise)
SNR_fail = -20; 
[P_BF_fail, P_MU_fail] = run_simulation(M, K, DOAs, N_snap, SNR_fail, d, lambda, theta_scan);

% Case B: SUCCESS (Clean signal)
SNR_success = 20; 
[P_BF_succ, P_MU_succ] = run_simulation(M, K, DOAs, N_snap, SNR_success, d, lambda, theta_scan);

% Plotting
create_comparison_plot('Effect of Signal-to-Noise Ratio (SNR)', theta_scan, ...
    P_BF_fail, P_MU_fail, M, SNR_fail, N_snap, ...
    P_BF_succ, P_MU_succ, M, SNR_success, N_snap, DOAs);


%% SIMULATION 6: EFFECT OF NUMBER OF SNAPSHOTS (N)
% Objective: Show that insufficient data leads to poor covariance estimation.
fprintf('Running Sim 6: Effect of Snapshots...\n');

% Common Parameters
K = 2; DOAs = [0 5]; % Closely spaced
M = 16; SNR = 10;

% Case A: FAIL (Insufficient data samples)
N_fail = 3; % Only 5 snapshots
[P_BF_fail, P_MU_fail] = run_simulation(M, K, DOAs, N_fail, SNR, d, lambda, theta_scan);

% Case B: SUCCESS (Converged covariance matrix)
N_success = 1000; 
[P_BF_succ, P_MU_succ] = run_simulation(M, K, DOAs, N_success, SNR, d, lambda, theta_scan);

% Plotting
create_comparison_plot('Effect of Snapshots (N)', theta_scan, ...
    P_BF_fail, P_MU_fail, M, SNR, N_fail, ...
    P_BF_succ, P_MU_succ, M, SNR, N_success, DOAs);


%% --- HELPER FUNCTIONS ---

function [P_BF_dB, P_MU_dB] = run_simulation(M, K, doas, N, SNR, d, lambda, scan)
    % 1. Generate Signals (Complex Random)
    S = (randn(K, N) + 1j*randn(K, N))/sqrt(2);
    
    % 2. Steering Matrix (True)
    A = zeros(M, K);
    k_wave = 2*pi/lambda;
    for i = 1:K
        A(:,i) = exp(-1j * k_wave * d * (0:M-1)' * sin(deg2rad(doas(i))));
    end
    
    % 3. Generate Data (X) with Noise
    X_sig = A * S;
    sig_power = mean(abs(X_sig(:)).^2);
    noise_power = sig_power / (10^(SNR/10));
    Noise = sqrt(noise_power/2) * (randn(M, N) + 1j*randn(M, N));
    X = X_sig + Noise;
    
    % 4. Covariance Matrix Estimation
    Rx = (X*X')/N;
    
    % 5. Beamforming Spectrum
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
    % Subspace separation
    if M > K
        En = E(:, K+1:end); % Noise subspace
        P_MU = zeros(size(scan));
        for i = 1:length(scan)
            a_theta = exp(-1j * k_wave * d * m_vec * sin(deg2rad(scan(i))));
            denom = sum(abs(En' * a_theta).^2);
            P_MU(i) = 1 / (denom + eps);
        end
    else
        % Fallback if M is too low for subspace methods
        P_MU = zeros(size(scan)); 
    end
    
    % Normalize
    P_BF_dB = 10*log10(P_BF / max(P_BF));
    P_MU_dB = 10*log10(P_MU / max(P_MU));
end

function create_comparison_plot(fig_title, scan, BF1, MU1, M1, S1, N1, BF2, MU2, M2, S2, N2, true_doas)
    figure('Position', [100, 100, 1200, 500], 'Name', fig_title);
    
    % --- LEFT SUBPLOT: FAIL CASE ---
    subplot(1, 2, 1);
    plot(scan, BF1, 'LineWidth', 1.2, 'DisplayName', 'Beamforming'); hold on;
    plot(scan, MU1, 'LineWidth', 1.5, 'DisplayName', 'MUSIC');
    for k = 1:length(true_doas)
        xline(true_doas(k), '--k', 'HandleVisibility', 'off', 'Alpha', 0.5);
    end
    grid on; legend('Location', 'northeast');
    xlabel('Angle (deg)'); ylabel('Spectrum (dB)');
    title(sprintf('M=%d, SNR=%d dB, N=%d', M1, S1, N1));
    ylim([-50 5]); xlim([min(true_doas)-40, max(true_doas)+40]);

    % --- RIGHT SUBPLOT: SUCCESS CASE ---
    subplot(1, 2, 2);
    plot(scan, BF2, 'LineWidth', 1.2, 'DisplayName', 'Beamforming'); hold on;
    plot(scan, MU2, 'LineWidth', 1.5, 'DisplayName', 'MUSIC');
    for k = 1:length(true_doas)
        xline(true_doas(k), '--k', 'HandleVisibility', 'off', 'Alpha', 0.5);
    end
    grid on; legend('Location', 'northeast');
    xlabel('Angle (deg)'); ylabel('Spectrum (dB)');
    title(sprintf('M=%d, SNR=%d dB, N=%d', M2, S2, N2));
    ylim([-50 5]); xlim([min(true_doas)-40, max(true_doas)+40]);
    
    sgtitle(fig_title);
end