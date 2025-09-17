function x = gausselim(A,b)
    Aaug = [A,b];
    n = size(A, 1);  
    for i = 1:n
        for j = i+1 :n
           Aaug(j,:) = Aaug(j,:)-(Aaug(j,i)/(Aaug(i,i))*Aaug(i,:));
        end
    end
    U = A;
    x = backsub(U, b);
end