function [L,U,swapCount] = LUdecomp(A)
    U = A;
    n = size(U, 1);  
    L = eye(n);
    swapCount = 0;

    for i = 1:n
        [~, pivot] = max(abs(U(i:end, i)));
        pivot = pivot + i - 1;

        if pivot ~= i
            U([i pivot], :) = U([pivot i], :);
            if i > 1
                L([i pivot], 1:i-1) = L([pivot i], 1:i-1);
            end
            swapCount = swapCount + 1;
        end

        for j = i+1 : n
            L(j,i) = U(j,i)/U(i,i);
            U(j,:) = U(j,:) - L(j,i)*U(i,:);
        end
    end
end
