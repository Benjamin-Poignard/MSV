function L = Loss_penalized_objective(Y,X,param,lambda,method)

% Inputs:
%         - Y: vector of response variables
%         - X: matrix of lagged variables
%         - param: vector of parameters of interest
%         - lambda: tuning parameter (user specified)
%         - method: 'scad' or 'mcp'

% Output:
%         - L: value of the objective function (OLS function)

switch method
    case 'scad'
        pen = scad(param,lambda,3.5);
    case 'mcp'
        pen = mcp(param,lambda,3);
end
L = sum((Y-X*param).^2)/(2*length(Y)) + pen;

