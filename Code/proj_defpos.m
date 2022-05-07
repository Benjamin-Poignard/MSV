function H = proj_defpos(M)

[P,~] = eig(M);
L = diag(subplus(eig(M)))+1000;
H = P*L*P';