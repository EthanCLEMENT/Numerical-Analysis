function d = mydet(A)
    [L,U,swapCount] = LUdecomp(A);
    diagProductU = 1;

    for i = 1:length(A)
        diagProductU = diagProductU * U(i,i);
    end

    d = (-1)^swapCount * diagProductU;
end
