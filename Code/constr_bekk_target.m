function [c,ceq] = constr_bekk_target(x)

c = [ -x+0.0001 ; 
    x-0.99];    
ceq = [];