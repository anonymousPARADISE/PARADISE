%% ============================================================
%  Run PARADISE on PEMS
% ============================================================

delete(gcp('nocreate'));

numW = 20;
N = maxNumCompThreads(numW);
p = parpool('threads');

%% Dataset setting
dataset_name = 'PEMS';
test_ratio = 0.3;   % 출력용

%% Hyperparameters for PEMS
R = 10;
lambda_u = 0.01;
lambda_l = 0.1;
lambda_s = 1;
lambda_t = 0.1;

patience = 5;
conv = 0.0001;
maxiter = 30;
epochs = 1;

%% Load data
load('./data/PEMS_train.mat', 'data');
X_train = data;

load('./data/PEMS_valid.mat', 'data');
X_valid = data;

load('./data/PEMS_test.mat', 'data');
X_test = data;

K = length(X_train);

%% Prepare missing indices
missing_ind = cell(1, K);
missing_ind_valid = cell(1, K);
missing_ind_test = cell(1, K);

for k = 1:K
    missing_ind_valid{k} = X_valid{k}(:, 1:3);
    missing_ind_test{k} = X_test{k}(:, 1:3);
    missing_ind{k} = [missing_ind_valid{k}; missing_ind_test{k}];
end

%% Remove NaN entries from validation/test data
for k = 1:K
    X_test{k}(:, 1:3) = int32(X_test{k}(:, 1:3));
    X_valid{k}(:, 1:3) = int32(X_valid{k}(:, 1:3));

    nan_mask = isnan(X_test{k}(:, 4));
    X_test{k}(nan_mask, :) = [];

    nan_mask = isnan(X_valid{k}(:, 4));
    X_valid{k}(nan_mask, :) = [];
end

%% Construct Fourier basis
frequency = dominant_frequency(X_train);

P_list = cellfun(@(r) r.period, frequency, 'UniformOutput', false);

Z = fourier_basis(X_train, P_list, 5, 1);

%% Train and evaluate PARADISE
fprintf('Dataset: %s\n', dataset_name);
fprintf('Hyperparameters: R=%d, lambda_u=%.2f, lambda_l=%.2f, lambda_s=%.2f, lambda_t=%.2f\n', ...
    R, lambda_u, lambda_l, lambda_s, lambda_t);

res_epochs = zeros(epochs, 1);
total_time = 0;

for epoch = 1:epochs
    [U, UT, US, S, V, fit_each, times] = PARADISE( ...
        X_train, X_valid, Z, R, missing_ind, maxiter, conv, patience, ...
        lambda_u, lambda_t, lambda_s, lambda_l);

    NRE_res = NRE(X_test, U, S, V, K);
    res_epochs(epoch) = NRE_res;

    fprintf('Epoch %d/%d | NRE: %.6f | missing ratio: %.1f | time per iteration: %.6f\n', ...
        epoch, epochs, NRE_res, test_ratio, times);

    total_time = total_time + times;
end

avg_time = total_time / epochs;

fprintf('Average NRE: %.6f\n', mean(res_epochs));
fprintf('Average runtime: %.6f\n', avg_time);