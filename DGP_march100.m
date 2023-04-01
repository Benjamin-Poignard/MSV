function [res,Sigma] = DGP_march100(T,N)

%%%%%% define the vectors and matrix %%%%%%
% Sigma is the true variance covariance matrix
Sigma = zeros(N,N,T); res = zeros(T,N); 
% initalization
res(1,:) = mvnrnd(zeros(N,1),eye(N)); 

% for simulating the DGP: stationarity conditions are necessary
cond = true;
while cond
    temp = round(0.75*rand(N)); temp(temp==0)=-1;
    for kk = 1:N
        for ii = kk+1:N
            temp(kk,ii) = temp(ii,kk);
        end
    end
    
    Omega = rand(N,N); Omega = temp.*(Omega*Omega'/8200);
    for ii = 1:N
        Omega(ii,ii) = 0.08+0.11*rand(1);
    end
    
    temp = round(0.75*rand(N^2)); temp(temp==0)=-1;
    B_temp = temp.*tril((0.05+(0.2-0.05)*rand(N^2))/48000,0);
    B = B_temp + B_temp';
    
    for ii = 1:N^2
        B(ii,ii) = 0.0048+(0.015-0.0048)*rand(1);
     end
     B_stat = positivity(B,N,'vech'); 
             
     quant1 = eig(Omega); quant2 = min(eig(B)); quant3 = max(abs(eig(B_stat)));
    
     cond = (any(quant1<0.000001)) || (quant2<0.0001) || (quant3>0.9999);

end
% when cond = 0, the true coefficients entering the BEKK dynamic satisfy
% the stationarity conditions

% simulation of the true variance covariance matrix and the observations
% res simulated in multivariate Gaussian distribution
for t = 2:T+1
    Sigma(:,:,t) = Omega + kron(eye(N),res(t-1,:))*B*kron(eye(N),res(t-1,:)');
    res(t,:) = mvnrnd(zeros(N,1),Sigma(:,:,t));
end
res = res(2:end,:); Sigma = Sigma(:,:,2:end); 