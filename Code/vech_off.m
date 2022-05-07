function M = vech_off(Mt,d)

M = tril(ones(d),-1);
M(M==1) = Mt;
M = M + M' + eye(d,d);

