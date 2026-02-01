% -------------------------------------------------------------------------
% INPUTS
% X: input slices
% Z: fixed sinusoid/Fourier basis
% R: rank
% missing_ind: missing/observed index metadata per slice
% maxiter, conv: maximum #iterations and early-stop tolerance
% lambda_u, lambda_t, lambda_s, lambda_l: regularization constants
%
% OUTPUTS
% U: temporal factor (trend + seasonal)
% S: slice weights
% V: shared column factor
% fit_each: per-iteration NRE on observed entries
% times: average update time per iteration (sec)
% -------------------------------------------------------------------------
function [U, UT, US, S, V, fit_each, times] = PARADISE(X_train, X_valid, Z, R, missing_ind, maxiter, conv, patience, lambda_u, lambda_t, lambda_s, lambda_l)

%% initialization
    best_valid_nre = inf;
    no_improvement_count = 0;

    ConvCrit = conv;
    flag = 0;
    fit_each = zeros(maxiter, 1);
    valid_nre_each = zeros(maxiter, 1);

    K = length(X_train);

    J = size(X_train{1}, 2);
    U = cell(K,1);
    H = rand(R,R);
    V = rand(J,R);
    W = rand(K,R);
    S = cell(K, 1);
    Q = cell(K, 1);
    UT = cell(K, 1);
    US = cell(K, 1);
    missing_ratio = zeros(K,1);
    parfor i = 1:K
        S{i}  = diag(rand(R, 1));
        Q{i} = randn(size(X_train{i},1), R); 
        UT{i} = randn(size(X_train{i}, 1), R);
        US{i} = randn(size(X_train{i}, 1), R);
        C{i} = randn(size(Z{i},2), R);
    end

    U_best = U; W_best = W; V_best = V;
    missing_ind_mat = cell(K,1);
    parfor k=1:K
        Ik = size(X_train{k}, 1);
        J = size(X_train{k}, 2);
        missing_ind_mat{k} = sparse(missing_ind{k}(:,2), missing_ind{k}(:,3), 1, Ik, J);
    end

    normX_train = 0;
    for k=1:length(X_train)
        normX_train = normX_train + norm(X_train{k}(find(missing_ind_mat{k} == 0)), "fro")^2;
    end

    recLoss = 0;
    NRE_valid = 0;
    iter = 0;
    times = 0;
    
    %% training
    for it=1:maxiter
        oldrec = recLoss;
        oldNRE = NRE_valid; 

        tic;    
        UT = updateUT(X_train, missing_ind_mat, UT, US, Q, W, V, H, lambda_u, lambda_t);
        US = updateUS(X_train, missing_ind_mat, UT, US, Q, W, V, H, Z, C, lambda_u, lambda_s);
        C = updateC(Z, US, lambda_s, lambda_l);
        W = updateW(X_train, missing_ind_mat, UT, US, V, W, lambda_l);
        V = updateV(X_train, missing_ind_mat, UT, US, V, W, lambda_l);
        [Q, H] = updateQH(H, UT, US);
        times = times + toc;


        parfor k = 1:K
            U{k} = UT{k} + US{k};
            S{k} = diag(W(k, :));
        end
        
        recLoss = 0;


        for k=1:K
            Ik = size(X_train{k}, 1);
            diff = X_train{k} - U{k} * diag(W(k,:)) * V';
            diff = diff(find(missing_ind_mat{k} == 0));
            recLoss = recLoss + norm(diff, "fro")^2;
        end
        NRE_train = recLoss / normX_train;

        fit_each(it) = NRE_train;

        NRE_valid = NRE(X_valid, U, S, V, K);
        valid_nre_each(it) = NRE_valid;

        if (NRE_valid + conv) < best_valid_nre
            best_valid_nre = NRE_valid;
            no_improvement_count = 0;
            U_best = U; W_best = W; V_best = V;
        else
            fprintf('there are no improvements...\n');
            no_improvement_count = no_improvement_count + 1;
        end

        fprintf('Iter %2d: Train NRE = %.4f, Valid NRE = %.4f (Best Valid NRE: %.4f)\n', ...
            it, NRE_train, NRE_valid, best_valid_nre);          
        iter = iter + 1;

        if no_improvement_count >= patience
            fprintf('Early stopping triggered\n');
            break
        end

    end
    
    U = U_best;
    V = V_best;
    
    parfor k=1:K
        S{k} = diag(W_best(k,:));
    end
    fit_each = fit_each(1:iter);
    times = times / iter;

   
end