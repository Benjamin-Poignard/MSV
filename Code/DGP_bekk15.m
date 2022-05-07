function [res,Sigma] = DGP_bekk15(T,N)

%%%%%% define the vectors and matrix %%%%%%
% Sigma is the true variance covariance matrix
Sigma = zeros(N,N,T); res = zeros(T,N); 
% initalization
res(1,:) = mvnrnd(zeros(N,1),eye(N,N)); 

% for simulating the DGP: stationarity conditions are necessary
cond = true;
D = full(DuplicationM(N)); D_p = (D'*D)\D';
while cond
    temp = round(0.75*rand(N)); temp(temp==0)=-1;
    for kk = 1:N
        for ii = kk+1:N
            temp(kk,ii) = temp(ii,kk);
        end
    end
    
    Omega = rand(N,N); Omega = temp.*(Omega*Omega'/300);
    for ii = 1:N
        Omega(ii,ii) = 0.08+0.1*rand(1);
    end
    
    %B = 0.25;
    temp = round(0.75*rand(N^2)); temp(temp==0)=-1;
    B = (-1+2*rand(N))/3.2; A = (-1+2*rand(N))/3.2;
    
    quant1 = eig(Omega); quant2 = max(abs(eig(D_p*(kron(A,A) + kron(B,B))*D)));
    
    cond = (any(quant1<0.000001)) || (quant2>0.9999);
end
% when cond = 0, the true coefficients entering the BEKK dynamic satisfy
% the stationarity conditions
cond

% simulation of the true variance covariance matrix and the observations
% res simulated in multivariate Gaussian distribution
for t = 2:T+1
    Sigma(:,:,t) = Omega + A*res(t-1,:)'*res(t-1,:)*A' + B*Sigma(:,:,t-1)*B';
    res(t,:) = mvnrnd(zeros(N,1),Sigma(:,:,t));
end
res = res(2:end,:); Sigma = Sigma(:,:,2:end); 