function [c,ceq] = dcc_constr(x,stdresid)

a = x(1); b = x(2);
Qbar=(stdresid'*stdresid).*(1/length(stdresid));
K=Qbar*(1-a-b);
E=eig(K);
eps=1e-3;
c = [];
c = -min(E);
k1 = a+b-0.99999;
k2 = -a+eps;
k3 = -b;
k4 = a-0.15;
k5 = b-0.999;
c = [c ; k1 ; k2 ; k3 ; k4 ; k5];
ceq = [];


