function x = solveAxb(A,b)
    n = size(A, 1);  
    
    for i = 1:n
        [~, pivot] = max(abs(A(i:end, i)));
        pivot = pivot + i - 1;
        A([i pivot], :) = A([pivot i], :);
        b([i pivot]) = b([pivot i]);
        
        for j = i+1 :n
            factor = A(j,i) / A(i,i);
            A(j,:) = A(j,:) - factor * A(i,:);
            b(j) = b(j) - factor * b(i);
        end
    end

    x = backsub(A,b);
end
