function [L, D] = ldlt(A)
    n = size(A, 1);
    L = eye(n);
    D = zeros(n);

    for i = 1:n
        sum_diag = 0;
        for k = 1:i-1
            sum_diag = sum_diag + L(i,k)^2 * D(k,k);
        end
        D(i,i) = A(i,i) - sum_diag;

        for j = i+1:n
            sum_sub_diag = 0;
            for k = 1:i-1
                sum_sub_diag = sum_sub_diag + L(j,k) * L(i,k) * D(k,k);
            end
            L(j,i) = (A(j,i) - sum_sub_diag) / D(i,i);
        end
    end
end
