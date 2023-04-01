function theta = VAR_penalized(Y,X,lambda,method)

%         - Y: vector of response variables
%         - X: matrix of lagged variables
%         - lambda: tuning parameter (user specified)
%         - method: 'lasso', 'alasso', 'scad' or 'mcp'
%           'alasso' stands for adaptive LASSO
%           'lasso' and 'alasso' are solved by the Shooting algorithm
%           'scad' and 'mcp' are numerically solved by fmincon

% Output:
%         - theta: vector of estimated parameters

switch method
    case 'lasso'
        theta = lassoShooting(Y,X,lambda,0);
    case 'alasso'
        theta = lassoShooting(Y,X,lambda,3);
    case 'scad'
        theta = scad_mcp_numerical(Y,X,lambda,method);
    case 'mcp'
        theta = scad_mcp_numerical(Y,X,lambda,method);
end