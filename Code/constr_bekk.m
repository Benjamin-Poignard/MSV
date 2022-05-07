function [c,ceq] = constr_bekk(x,N)

if size(x,2)>1
   x = x'; 
end
x_c = x(1:N*(N+1)/2); x = x(N*(N+1)/2+1:end);
c = [ 
    0.000000001-min(eig(dvech(x_c,N)));
    sum(x)-0.999;
    0.00001-x];
ceq = [];