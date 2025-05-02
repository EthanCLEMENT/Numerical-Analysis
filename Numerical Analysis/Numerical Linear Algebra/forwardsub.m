function y = forwardsub(L,b)
    for i = 1: length(L)
        y(i) = b(i);
        sum_ax = 0;
        for j = 1 :i-1
            sum_ax = sum_ax + L(i,j)*y(j);
        end
        y(i) = b(i) - sum_ax;

    end
end