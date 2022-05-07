function [parameters,Rt,H]=dcc_mvgarch(data)

[t,k]=size(data);
archP = 1; garchQ = 1; 

optimoptions.Hessian = 'bfgs';
optimoptions.MaxRLPIter = 1000;
optimoptions.Algorithm = 'sqp';
optimoptions.TolCon = 1e-09;
optimoptions.TolRLPFun = 1e-09;
optimoptions.MaxSQPIter = 1000;
for i=1:k
    [univariate{i}.parameters, univariate{i}.likelihood, univariate{i}.stderrors, univariate{i}.robustSE, univariate{i}.ht, univariate{i}.scores] ...
        = fattailed_garch(data(:,i) , archP , garchQ , 'NORMAL',[], optimoptions);
    stdresid(:,i)=data(:,i)./sqrt(univariate{i}.ht);
end

optimoptions.Hessian = 'bfgs';
optimoptions.MaxRLPIter = 5000;
optimoptions.Algorithm = 'sqp';
optimoptions.TolCon = 1e-09;
optimoptions.TolRLPFun = 1e-09;
optimoptions.MaxSQPIter = 8000;
optimoptions.Diagnostics = 'on';
optimoptions.Jacobian = 'off';
optimoptions.Display = 'iter';

condi = true;
while condi
    dccstarting(1) = 0.0001+(0.01-0.001)*rand(1,1);
    dccstarting(2) = 0.6+(0.92-0.6)*rand(1,1);
    [c,~] = dcc_constr(dccstarting,stdresid);
    condi = any(c>0);
end

[dccparameters,~,~,~,~,~]=fmincon(@(x)dcc_mvgarch_likelihood(x,stdresid),dccstarting,[],[],[],[],[],[],@(x)dcc_constr(x,stdresid),optimoptions);

% estimated dcc parameters
parameters=[];
H=zeros(t,k);
for i=1:k
    parameters=[parameters;univariate{i}.parameters];
    H(:,i)=univariate{i}.ht;
end
parameters=[parameters;dccparameters'];

% Generate the correlation matrix process
[~, Rt, ~, ~]=dcc_mvgarch_full_likelihood(parameters, data);
