function H = proj_defpos(M)

[P,~] = eig(M);
L = diag(subplus(eig(M)))+0.001;
H = P*L*P';