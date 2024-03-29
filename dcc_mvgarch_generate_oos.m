function [logL, Rt, likelihoods, Qt]=dcc_mvgarch_generate_oos(parameters,data,data_in,h_in)

% Out-of-sample full log-likelihood of the scalar DCC process with 
% parameters estimated in-sample

% Inputs:
%        - parameter: 3*N + 2 x 1 parameter vector of interest (includes
%        the univariate GARCH(1,1) parameters and scalar DCC parameters)
%        - data_out: T_out x N matrix of obversations, out-of-sample, with
%        T_out the length of the out-of-sample period
%        - data_in: T_in x N matrix of observations, in-sample, with T_in
%        the length of the in-sample period
%        - h_in: T_in x N vector of univariate conditional variance
%        processes, in-sample

% Outputs:
%        - logL: composite log-likelihood value evaluated at parameter
%        -  Rt: N x N x T_out correlation process generated from the scalar
%        DCC
%        - likelihoods: T_out x 1 vector of log-likelihood evaluated at
%        parameter such that logL = sum(likelihoods)
%        - Qt: N x N x T_out process of the underlying Qt matrix generated 
%        from the scalar DCC model

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