%% Array-Based DOA Estimation Using Digital Signal Processing Techniques
% EE473 Term Project
% Alperen Kahraman & Abdulhalim Kiraz

clear; clc; close all;

%% Global System Parameters
c = 3e8; % Speed of light
fc = 2.4e9; % Carrier frequency
lambda = c/fc; % Wavelength
d = lambda/2; % Distance between antennas
theta_scan = -90:0.1:90;  % Scan interval
M_default = 16; % Antenna number
N_snap_default = 1000; % Snapshot number

%% Functions for algorithms and simulations

% Main function to create simulations
function [P_BF_dB, P_MU_dB] = run_simulation(M, K, doas, N, SNR, d, lambda, scan)
    S = (randn(K, N) + 1j*randn(K, N))/sqrt(2);
    A = steering_matrix_ula(doas, M, d, lambda);
    X = generate_data(A, S, SNR);
    Rx = (X * X') / N;
    
    % Compute spatial spectrum for both methods
    P_BF = conventional_bf(Rx, M, d, lambda, scan);
    if M > K
        P_MU = music_spectrum(Rx, K, d, lambda, scan);
    else
        P_MU = zeros(size(scan));
    end
    
    % Convert to normalized dB scale for plotting
    P_BF_dB = normalize_db_eps(P_BF);
    P_MU_dB = normalize_db_eps(P_MU);
end

% Conventional Beamforming implementation
function P_BF = conventional_bf(Rx, M, d, lambda, theta_scan)
    k = 2*pi / lambda;
    m_vec = (0:M-1).';
    P_BF = zeros(size(theta_scan));
    for i = 1:length(theta_scan)
        a = exp(-1j * k * d * m_vec * sin(theta_scan(i) * pi / 180));
        w = a / M;
        P_BF(i) = real(w' * Rx * w);
    end
end

% MUSIC Implementation
function P_MU = music_spectrum(Rx, K, d, lambda, theta_scan)
    [M_curr, ~] = size(Rx); 
    [E, D] = eig(Rx);
    [~, idx] = sort(diag(D), 'descend');
    En = E(:, idx(K+1:end)); % selection of the noise eigenvectors
    k = 2*pi / lambda;
    m_vec = (0:M_curr-1).';
    P_MU = zeros(size(theta_scan));
    for i = 1:length(theta_scan)
        a = exp(-1j * k * d * m_vec * sin(theta_scan(i) * pi / 180));
        P_MU(i) = 1 / (sum(abs(En' * a).^2) + eps);
    end
end

% Forward Backward Spatial Smoothing Implementation
function Rx_ss = forward_backward_spatial_smoothing(X, M_sub)
    [M, N_snap] = size(X);
    L = M - M_sub + 1; 
    Rx_fwd = zeros(M_sub, M_sub);
    for i = 1:L
        X_sub = X(i : i+M_sub-1, :);
        Rx_fwd = Rx_fwd + (X_sub * X_sub') / N_snap;
    end
    Rx_fwd = Rx_fwd / L;
    J = fliplr(eye(M_sub)); % Exchange matrix for backward smoothing
    Rx_ss = (Rx_fwd + J * conj(Rx_fwd) * J) / 2;
end

% Steering matrix construction
function A = steering_matrix_ula(doas_deg, M, d, lambda)
    k = 2*pi / lambda;
    m_vec = (0:M-1).';
    A = zeros(M, length(doas_deg));
    for idx = 1:length(doas_deg)
        A(:, idx) = exp(-1j * k * d * m_vec * sin(doas_deg(idx) * pi / 180)); 
    end
end

% Definition of X matrix including the noise
function X = generate_data(A, S, SNR_dB)
    [M, ~] = size(A); [~, Nsnap] = size(S);
    X_sig = A * S;
    noise_power = mean(abs(X_sig(:)).^2) / (10^(SNR_dB/10));
    X = X_sig + sqrt(noise_power/2) * (randn(M, Nsnap) + 1j*randn(M, Nsnap));
end

%% Visual Formatting Helpers 

function create_comparison_plot(title_str, scan, BF1, MU1, M1, S1, N1, BF2, MU2, M2, S2, N2, true_doas)
    figure('Name', title_str);
    subplot(1, 2, 1);
    plot(scan, BF1, 'LineWidth', 1.2); hold on; plot(scan, MU1, 'LineWidth', 1.5);
    for k = 1:length(true_doas), xline(true_doas(k), '--k', 'Alpha', 0.5); end
    grid on; legend('Beamforming','MUSIC'); title(sprintf('M=%d, SNR=%d dB, N=%d', M1, S1, N1));
    ylim([-50 5]); xlim([min(true_doas)-40, max(true_doas)+40]);
    
    subplot(1, 2, 2);
    plot(scan, BF2, 'LineWidth', 1.2); hold on; plot(scan, MU2, 'LineWidth', 1.5);
    for k = 1:length(true_doas), xline(true_doas(k), '--k', 'Alpha', 0.5); end
    grid on; legend('Beamforming','MUSIC'); title(sprintf('M=%d, SNR=%d dB, N=%d', M2, S2, N2));
    ylim([-50 5]); xlim([min(true_doas)-40, max(true_doas)+40]);
    sgtitle(title_str);
end

function plot_results(scan, BF, MU, true_doas, prefix, K, M)
    figure();
    plot(scan, BF, 'LineWidth', 1.5); hold on; plot(scan, MU, 'LineWidth', 1.5);
    for k = 1:length(true_doas), xline(true_doas(k), '--k', 'LineWidth', 1); end
    grid on; legend('Beamforming','MUSIC');
    xlabel('Angle (Degrees)'); ylabel('Normalized Spectrum (dB)');
    title(sprintf('%s: K=%d, DOAs=[%s], M=%d', prefix, K, num2str(true_doas), M));
    xlim([min(true_doas)-30, max(true_doas)+30]);
end

% normalization
function P_dB = normalize_db_eps(P)
    P_dB = 10*log10(P / max(P) + eps);
end

% Plotting the true DOAs
function add_true_doa_lines(doas, K)
    yL = ylim;
    for k = 1:K, plot([doas(k) doas(k)], yL, 'k--'); end
end


%% Part 1: Simple Cases

% Simulation 1: Single Source Case 
K1 = 1; DOAs1 = [10]; SNR1 = 10;
[P_BF1, P_MU1] = run_simulation(M_default, K1, DOAs1, N_snap_default, SNR1, d, lambda, theta_scan);
plot_results(theta_scan, P_BF1, P_MU1, DOAs1, 'Simulation 1', K1, M_default);

%  Simulation 2: Two Sources Case 
K2 = 2; DOAs2 = [-10 30]; SNR2 = 10;
[P_BF2, P_MU2] = run_simulation(M_default, K2, DOAs2, N_snap_default, SNR2, d, lambda, theta_scan);
plot_results(theta_scan, P_BF2, P_MU2, DOAs2, 'Simulation 2', K2, M_default);

%  Simulation 3: Small Separation Case 
K3 = 2; DOAs3 = [10 15]; SNR3 = 10;
[P_BF3, P_MU3] = run_simulation(M_default, K3, DOAs3, N_snap_default, SNR3, d, lambda, theta_scan);
plot_results(theta_scan, P_BF3, P_MU3, DOAs3, 'Simulation 3', K3, M_default);

%% Part 2: Effects of Parameters

%  Simulation 4: Effect of Antenna Number (M) 
DOAs4 = [10 20];
[P_BF4_L, P_MU4_L] = run_simulation(4, 2, DOAs4, 1000, 10, d, lambda, theta_scan);
[P_BF4_H, P_MU4_H] = run_simulation(16, 2, DOAs4, 1000, 10, d, lambda, theta_scan);
create_comparison_plot('Effect of Antenna Number (M)', theta_scan, ...
    P_BF4_L, P_MU4_L, 4, 10, 1000, P_BF4_H, P_MU4_H, 16, 10, 1000, DOAs4);

%  Simulation 5: Effect of SNR (dB) 
DOAs5 = [-10 15];
[P_BF5_L, P_MU5_L] = run_simulation(16, 2, DOAs5, 1000, -20, d, lambda, theta_scan);
[P_BF5_H, P_MU5_H] = run_simulation(16, 2, DOAs5, 1000, 20, d, lambda, theta_scan);
create_comparison_plot('Effect of SNR (dB)', theta_scan, ...
    P_BF5_L, P_MU5_L, 16, -20, 1000, P_BF5_H, P_MU5_H, 16, 20, 1000, DOAs5);

%  Simulation 6: Effect of Snapshots (N) 
DOAs6 = [0 5];
[P_BF6_L, P_MU6_L] = run_simulation(16, 2, DOAs6, 3, 10, d, lambda, theta_scan);
[P_BF6_H, P_MU6_H] = run_simulation(16, 2, DOAs6, 1000, 10, d, lambda, theta_scan);
create_comparison_plot('Effect of Snapshots (N)', theta_scan, ...
    P_BF6_L, P_MU6_L, 16, 10, 3, P_BF6_H, P_MU6_H, 16, 10, 1000, DOAs6);

%% Part 3: Forward-Backward Spatial Smoothing Algorithm

% Setup for Coherent Sources
K_fbss = 2; doas_fbss = [-10 5]; theta_scan_fbss = -90:0.5:90;
A_fbss = steering_matrix_ula(doas_fbss, M_default, d, lambda);

% Generation of Coherent Data
s_single = (randn(1, N_snap_default) + 1j*randn(1, N_snap_default))/sqrt(2);
S_coherent = repmat(s_single, K_fbss, 1); 
X_coh = generate_data(A_fbss, S_coherent, 10);
Rx_coh = (X_coh * X_coh') / N_snap_default;

% Standard MUSIC
P_MU_fail = normalize_db_eps(music_spectrum(Rx_coh, K_fbss, d, lambda, theta_scan_fbss));

% SS-MUSIC
M_sub = 7;
Rx_smoothed = forward_backward_spatial_smoothing(X_coh, M_sub);
P_MU_smoothed = normalize_db_eps(music_spectrum(Rx_smoothed, K_fbss, d, lambda, theta_scan_fbss));

figure('Name','Coherent Sources: MUSIC vs SS-MUSIC');

% Standard MUSIC plot
subplot(2,1,1);
plot(theta_scan_fbss, P_MU_fail, 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 1.2); hold on;
add_true_doa_lines(doas_fbss, K_fbss);
grid on; title('Coherent Signals with Standard MUSIC'); legend('Standard MUSIC', 'True DOAs');

% SS-MUSIC plot
subplot(2,1,2);
plot(theta_scan_fbss, P_MU_smoothed, 'Color', [0, 0.4470, 0.7410], 'LineWidth', 1.2); hold on;
add_true_doa_lines(doas_fbss, K_fbss);
grid on; title(['Forward-Backward Spatial Smoothing (Subarray Size: ' num2str(M_sub) ')']); legend('SS-MUSIC', 'True DOAs');