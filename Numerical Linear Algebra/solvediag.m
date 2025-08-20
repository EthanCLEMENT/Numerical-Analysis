function x = solvediag(A,b)
    x = zeros(length(b),1);
    for i = 1:length(A)
        x(i) = b(i)/A(i,i);
    end
end


