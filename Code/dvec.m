function M = dvec(K,N)

M = zeros(N,N);
for ii = 1:N
   M(:,ii) = K(N*(ii-1)+1:N*ii);
end