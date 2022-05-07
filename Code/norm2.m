function distance = norm2(R1,R2,N)

% N est la dimension de la matrice
distance = 0;
for i = 1:N
    for j = 1:N
        distance = distance + (abs(R1(i,j) - R2(i,j)))^2;
    end    
end

distance = sqrt(distance);