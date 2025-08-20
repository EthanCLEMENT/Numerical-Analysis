function x = cholsekydecomp(A, b)
    n = length(A);
    L = zeros(n);

    for i = 1:n
        sum_diag = 0;
        for k = 1:i-1
            sum_diag = sum_diag + L(i,k)^2;
        end
        L(i,i) = sqrt(A(i,i) - sum_diag);

        for j = i+1:n
            sum_off_diag = 0;
            for k = 1:i-1
                sum_off_diag = sum_off_diag + L(j,k) * L(i,k);
            end
            L(j,i) = (A(j,i) - sum_off_diag) / L(i,i);
        end
    end

    y = forwardsub(L, b);
    x = backsub(L', y);
end
