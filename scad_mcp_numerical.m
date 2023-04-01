function theta = scad_mcp_numerical(Y,X,lambda,method)

% Inputs:
%         - Y: vector of response variables
%         - X: matrix of lagged variables
%         - lambda: tuning parameter (user specified)
%         - method: 'scad' or 'mcp', numerically solved by fmincon

% Output:
%         - theta: vector of estimated parameters

optimoptions.Hessian = 'bfgs';
optimoptions.MaxRLPIter = 300000;
optimoptions.MaxFunEvals = 300000;
optimoptions.Algorithm = 'sqp';
optimoptions.TolCon = 1e-09;
optimoptions.TolRLPFun = 1e-09;
optimoptions.MaxSQPIter = 300000;
optimoptions.Diagnostics = 'on';
optimoptions.Jacobian = 'off';
optimoptions.Display = 'iter';

param_init = (X'*X)\(X'*Y);
[theta,~,~,~,~,~]=fmincon(@(param)Loss_penalized_objective(Y,X,param,lambda,method),param_init,[],[],[],[],[],[],[],optimoptions);
theta(abs(theta)<0.0001)=0;