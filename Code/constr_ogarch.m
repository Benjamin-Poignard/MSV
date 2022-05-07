function [c,ceq] = constr_ogarch(x)

if size(x,2)>1
   x = x'; 
end
c = [ sum(x)-0.999;
    0.001-x;
    x-0.999];
ceq = [];