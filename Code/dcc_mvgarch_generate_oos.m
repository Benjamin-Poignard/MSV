function [logL, Rt, likelihoods, Qt]=dcc_mvgarch_generate_oos(parameters,data,data_in,h_in)

[t,k]=size(data);
index=1;
H=zeros(size(data));

for i=1:k
    univariateparameters=parameters(index:index+1+1);
    [~, H(:,i)] = dcc_univariate_simulate(univariateparameters,1,1,data(:,i));
    index=index+1+1+1;
end

stdresid=data./sqrt(H);

stdresid=[ones(1,k);stdresid];
a=parameters(index:index);
b=parameters(index+1:index+1);

Qbar=cov(data_in./sqrt(h_in));
Qt=zeros(k,k,t+1);
Qt(:,:,1)=repmat(Qbar,[1 1 1]);
Rt=zeros(k,k,t+1);
logL=0;
likelihoods=zeros(t+1,1);
H=[zeros(1,k);H];
for j=2:t+1
    Qt(:,:,j)=Qbar*(1-a-b);
    Qt(:,:,j)=Qt(:,:,j)+a*(stdresid(j-1,:)'*stdresid(j-1,:));
    Qt(:,:,j)=Qt(:,:,j)+b*Qt(:,:,j-1);
    Rt(:,:,j)=Qt(:,:,j)./(sqrt(diag(Qt(:,:,j)))*sqrt(diag(Qt(:,:,j)))');
end
Rt=Rt(:,:,(2:t+1));