function [parameters, loglikelihood, Ht, likelihoods, stdresid, A, B] = scalar_bekk_mvgarch(data)

p = 1; q = 1;
if size(data,2) > size(data,1)
    data=data';
end

[t k]=size(data);
garchmat=zeros(k,1+p+q);
optimoptions.Hessian = 'bfgs';
optimoptions.MaxRLPIter = 1000;
optimoptions.Algorithm = 'sqp';
optimoptions.TolCon = 1e-09;
optimoptions.TolRLPFun = 1e-09;
optimoptions.MaxSQPIter = 1000;
optimoptions.Display = 'iter';

for i=1:k
    temparam=fattailed_garch(data(:,i),p,q,'NORMAL',[],optimoptions);
    garchmat(i,:)=temparam';
end

A=mean(garchmat(:,2:p+1));
B=mean(garchmat(:,p+2:p+q+1));


optimoptions.Hessian = 'bfgs';
optimoptions.MaxRLPIter = 5000;
optimoptions.Algorithm = 'sqp';
optimoptions.TolCon = 1e-09;
optimoptions.TolRLPFun = 1e-09;
optimoptions.MaxSQPIter = 8000;
optimoptions.Diagnostics = 'on';
optimoptions.Jacobian = 'off';
optimoptions.Display = 'iter';

C=cov(data);
alpha0=sqrt(A);
beta0=sqrt(B);

StartC=C*(1-sum(alpha0.^2)-sum(beta0.^2));
CChol=chol(StartC)';
%warning off %#ok<WNOFF>
startingparameters=[vech(CChol);alpha0;beta0];

k2=k*(k+1)/2;

[parameters,~,~,~,~,~] = fmincon(@(x)scalar_bekk_mvgarch_likelihood(x,data,p,q,k,k2,t),startingparameters,[],[],[],[],[],[],@(x)constr_bekk(x,k),optimoptions);

[loglikelihood,likelihoods,Ht]=scalar_bekk_mvgarch_likelihood(parameters,data,p,q,k,k2,t);
loglikelihood=-loglikelihood;
likelihoods=-likelihoods;

stdresid=zeros(size(data));
for i=1:t
    stdresid(i,:)=data(i,:)*Ht(:,:,i)^(-0.5);
end
