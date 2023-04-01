function M = dvech(Mt,d)

M = tril(ones(d),0);
M(M==1) = Mt;
M = tril(M,-1)' + M;
