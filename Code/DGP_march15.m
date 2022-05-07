function [res,Sigma] = DGP_march15(T,N)

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
    
    Omega = rand(N,N); Omega = temp.*(Omega*Omega'/300);
    for ii = 1:N
        Omega(ii,ii) = 0.08+0.1*rand(1);
    end
    
    temp = round(0.75*rand(N^2)); temp(temp==0)=-1;
    B_temp = temp.*tril((0.2+(0.5-0.2)*rand(N^2))/550,0);
    B = B_temp + B_temp';
    
    temp = round(0.75*rand(N^2)); temp(temp==0)=-1;
    B_temp = temp.*tril((0.01+(0.2-0.01)*rand(N^2))/3850,0);
    B2 = B_temp + B_temp';
    
    
    for ii = 1:N^2
        B(ii,ii) = 0.05+(0.08-0.05)*rand(1);
        B2(ii,ii) = 0.001+(0.005-0.001)*rand(1);
     end
     B_stat = positivity(B,N,'vech'); B_stat2 = positivity(B2,N,'vech' );
             
     quant1 = eig(Omega); quant2 = min(eig(B)); quant3 = min(eig(B2)); quant4 = max(abs(eig(B_stat+B_stat2)));
    
     cond = (any(quant1<0.000001)) || (quant2<0.0001) || (quant3<0.0001) || (quant4>0.9999);
end

% simulation of the true variance covariance matrix and the observations
% res simulated in multivariate Gaussian distribution
for t = 2:T+1
    Sigma(:,:,t) = Omega + kron(eye(N),res(t-1,:))*B*kron(eye(N),res(t-1,:)');
    res(t,:) = mvnrnd(zeros(N,1),Sigma(:,:,t));
end
res = res(2:end,:); Sigma = Sigma(:,:,2:end); 