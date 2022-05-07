function [logL, Rt, likelihoods, Qt]=dcc_mvgarch_likelihood(params,stdresid)

[T,N]=size(stdresid);
a=params(1);
b=params(2);

Qbar=cov(stdresid);
Qt=zeros(N,N,T+1);
Rt=zeros(N,N,T+1);
Qt(:,:,1)=Qbar;
logL=0;
likelihoods=zeros(1,T+1);
stdresid=[zeros(1,N);stdresid];
for t=2:T+1
    Qt(:,:,t)=Qbar*(1-a-b) + a*(stdresid(t-1,:)'*stdresid(t-1,:)) + b*Qt(:,:,t-1);
    Rt(:,:,t)=Qt(:,:,t)./(sqrt(diag(Qt(:,:,t)))*sqrt(diag(Qt(:,:,t)))');
    likelihoods(t)=log(det(Rt(:,:,t)))+stdresid(t,:)*inv(Rt(:,:,t))*stdresid(t,:)';
    logL=logL+likelihoods(t);
end;

Qt=Qt(:,:,(2:T+1));
Rt=Rt(:,:,(2:T+1));
logL=(1/2)*logL;
likelihoods=(1/2)*likelihoods(2:T+1);