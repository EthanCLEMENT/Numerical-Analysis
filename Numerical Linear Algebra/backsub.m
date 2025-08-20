function x = backsub(A,b)
    x = zeros(length(b),1);
    for i = length(A):-1:1
        sum_ax = 0;
  
        for j = i+1: length(A)
            sum_ax = sum_ax + A(i,j) * x(j);
        end
        x(i) = (b(i) - sum_ax)/A(i,i);
    end
end