% Compute normalized reconstruction error on test entries.
%
% INPUT:
% data: test indices & values per slice
% U, S, V: factor matrices U, S, and V
% numK: number of slices 
%
% OUTPUT
% NRE_result: scalar normalized error 
function [MSE_result, MAE_result] = MSE(data, U, S, V, numK)
    indices = data(:, 1:3);
    vals = data(:,4);

    
    error_vals = zeros(numK,1);
    base = zeros(numK, 1);

    sq_error_sum = 0;
    abs_error_sum = 0;
    total_count = 0;

    for k = 1:numK
        k_indices = data{k}(:, 1:3);
        k_vals = data{k}(:, 4);

        recon = U{k} * S{k} * V';
        idx = sub2ind(size(recon), k_indices(:,2), k_indices(:,3));
        recon_vals = recon(idx);

        diff = k_vals - recon_vals;

        sq_error_sum  = sq_error_sum  + sum(diff.^2);
        abs_error_sum = abs_error_sum + sum(abs(diff));
        total_count   = total_count   + length(diff);
    end

    MSE_result = sq_error_sum / total_count;
    MAE_result = abs_error_sum / total_count;

end
