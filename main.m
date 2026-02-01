delete(gcp('nocreate'))
numW = 20;
N = maxNumCompThreads(numW);
p = parpool('threads');
patience = 5;

%% Load data
load_train_path = sprintf('./data/PEMS_train.mat');
load(sprintf(load_train_path, 'data'));
X_train = data;
load_valid_path = sprintf('./data/PEMS_valid.mat');
load(sprintf(load_valid_path, 'data'));
X_valid = data;
load_test_path = sprintf('./data/PEMS_test.mat');
load(sprintf(load_test_path), 'data');
X_test = data;
K = length(X_train);

%% Prepare missing indices (metadata of observed entries) --------------
missing_ind_test = cell(1,K);
missing_ind_valid = cell(1, K);
for k=1:K
    missing_ind_valid{k} = X_valid{k}(:, 1:3);
    missing_ind_test{k} = X_test{k}(:, 1:3);  
    missing_ind{k} = [missing_ind_valid{k};missing_ind_test{k}];
end  

for k=1:length(X_test)
    cast(X_test{k}(:,1:3), 'int32');
    cast(X_valid{k}(:, 1:3), 'int32');
    X_test{k}(:,1:3) = X_test{k}(:,1:3);
    nan_mask = isnan(X_test{k}(:, 4));
    X_test{k}(nan_mask, :) = [];

    X_valid{k}(:, 1:3) = X_valid{k}(:, 1:3);
    nan_mask = isnan(X_valid{k}(:, 4));
    X_valid{k}(nan_mask, :) = [];
end

frequency = dominant_frequency(X_train);
P_list = cellfun(@(r) r.period, frequency, 'UniformOutput', false);   
Z = fourier_basis(X_train, P_list, 5, 1);
%% Set hyperparameters
conv = 0.0001; maxiter = 30;
R = 10; lambda_u = 0.01; lambda_l = 10; lambda_s = 1; lambda_t = 0.1;
%% train and evaluate model
[U, UT, US, S, V, fit_each, times] = PARADISE(X_train, X_valid, Z, R, missing_ind, maxiter, conv, patience, lambda_u, lambda_t, lambda_s, lambda_l);
NRE_res = NRE(X_test, U, S, V, length(X_test));
fprintf('Proposed method fitness : %4f, time per iteration : %4f\n', NRE_res, times);  
