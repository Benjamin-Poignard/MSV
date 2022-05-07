% Simulation of a N-dimensional random vector denoted as 'res', where the
% data generating process (DGP) is based on a MGARCH-based dynamic:
% - a BEKK based process
% - a MARCH based process
% Each of these DGP is adapted depending on the dimension size to satisfy
% the stationarity conditions and positive definiteness constraints

%% MARCH based DGP; dimension N = 15
% this part of the code can easily be handled with a standard computer
% desktop
clear; clc;

% N: problem dimension; T: sample size
N = 15; T = 800;

% Nsim: number of batches
Nsim = 1; check = zeros(Nsim,5);

for oo = 1:Nsim
     
    % Generation of the MARCH based process
    [res,Sigma] = DGP_march15(T,N);
    
    % len is the size of the test set for cross-validation 
    len = T/2; 
    % fold is the number of folds for cross-validation
    folds = 4;
    
    % SCAD penalisation
    % p: number of specified lags
    % sqrt(log(p*N^2)/T): cross-validation is performed with a grid around
    % the value sqrt(log(p*N^2)/T), which is a standard tuning parameter
    % value
    % the choice of the grid is important to control for sparsity: the
    % larger, to more zeros are obtained
    p = 10;
    lambda = 0.2*(0.01:0.01:0.5)*sqrt(log(p*N^2)/T); 
    [H_scad,b_scad,~,~,~,~] = SV_process_memo(res,p,'scad','no-constant',lambda,len,folds);
    
    % MCP penalisation
    lambda = 0.2*(0.01:0.01:0.5)*sqrt(log(p*N^2)/T); 
    [H_mcp,b_mcp,~,~,~,~] = SV_process_memo(res,p,'mcp','no-constant',lambda,len,folds);
    
    % Non-penalised MSV 
    [H_npen,b_nonpen,~,~,~,~] = SV_process_memo(res,p,'nonpen','no-constant',lambda,len,folds);
    
    % scalar DCC, two-step Gaussian QMLE method
    res_temp = res(p+1:end,:); T = length(res_temp);
    [parameters_dcc,Rt]=dcc_mvgarch(res_temp);
    h_vol=zeros(length(res_temp),size(res_temp,2));
    index = 1;
    for jj=1:size(res_temp,2)
        univariateparameters=parameters_dcc(index:index+1+1);
        [simulatedata, h_vol(:,jj)] = dcc_univariate_simulate(univariateparameters,1,1,res_temp(:,jj));
        index=index+1+1+1;
    end
    clear jj
    h_vol = sqrt(h_vol);
    [~,Rt_dcc,~,~]=dcc_mvgarch_full_likelihood(parameters_dcc,res_temp);
    Hdcc = zeros(N,N,length(res_temp));
    for t = 1:T
        Hdcc(:,:,t) = diag(h_vol(t,:))*Rt_dcc(:,:,t)*diag(h_vol(t,:));
    end
    
    % CCC model, where the correlation matrix is evaluated as the sample
    % correlation, the marginals are univariate GARCH processes
    Hccc = zeros(N,N,length(res_temp)); R_ccc = corr(res_temp);
    for t = 1:T
        Hccc(:,:,t) = diag(h_vol(t,:))*R_ccc*diag(h_vol(t,:));
    end
    
    % Compare the true variance covariance Sigma and the estimated H using 
    % the distance criterion norm2 (Frobenius norm)
    % The lower the better
    
    T = length(res(p+1:end,:)); H_true = Sigma(:,:,p+1:end);
    distdcc = zeros(T,1); distccc = zeros(T,1);
    dist_scad = zeros(T,1); dist_mcp = zeros(T,1);
    dist_npen = zeros(T,1);
    for tt = 1:T
        dist_scad(tt) = norm2(H_true(:,:,tt),H_scad(:,:,tt),N);
        dist_npen(tt) = norm2(H_true(:,:,tt),H_npen(:,:,tt),N);
        dist_mcp(tt) = norm2(H_true(:,:,tt),H_mcp(:,:,tt),N);
        distdcc(tt) = norm2(H_true(:,:,tt),Hdcc(:,:,tt),N);
        distccc(tt) = norm2(H_true(:,:,tt),Hccc(:,:,tt),N);
    end
    
    % Compare the average error
    check(oo,:) = [mean(distdcc),mean(distccc),mean(dist_npen),mean(dist_scad),mean(dist_mcp)];

end
%% MARCH based DGP; N = 50
% this part of the code can be carried out on a standard computer, but the
% time estimation+generation can take a few hours for one run
% the experiments for dimensions>=50 were launched on a cluster (Intel(R)-Xeon(R) with CPU E7-8891 v3, 2.80GHz, 1.2 Terad byte)
clear; clc;
Nsim = 100; check = zeros(Nsim,5);
N = 50; T = 800;

for oo = 1:Nsim
    
    [res,Sigma] = DGP_march50(T,N);
    
    len = T/2; folds = 4;
 
    p = 5;
    lambda = 0.2*(0.01:0.01:0.5)*sqrt(log(p*N^2)/T);  
    [H_scad,b_scad,~,~,~,~] = SV_process_memo(res,p,'scad','no-constant',lambda,len,folds);
    
    [H_mcp,b_mcp,~,~,~,~] = SV_process_memo(res,p,'mcp','no-constant',lambda,len,folds); 
    
    [H_npen,b_nonpen,~,~,~,~] = SV_process_memo(res,p,'nonpen','no-constant',lambda,len,folds);
    
    res_temp = res(p+1:end,:); T = length(res_temp);
    [parameters_dcc,Rt]=dcc_mvgarch(res_temp);
    h_vol=zeros(length(res_temp),size(res_temp,2));
    index = 1;
    for jj=1:size(res_temp,2)
        univariateparameters=parameters_dcc(index:index+1+1);
        [simulatedata, h_vol(:,jj)] = dcc_univariate_simulate(univariateparameters,1,1,res_temp(:,jj));
        index=index+1+1+1;
    end
    clear jj
    h_vol = sqrt(h_vol);
    [~,Rt_dcc,~,~]=dcc_mvgarch_full_likelihood(parameters_dcc,res_temp);
    Hdcc = zeros(N,N,length(res_temp));
    for t = 1:T
        Hdcc(:,:,t) = diag(h_vol(t,:))*Rt_dcc(:,:,t)*diag(h_vol(t,:));
    end
    
    Hccc = zeros(N,N,length(res_temp)); R_ccc = corr(res_temp);
    for t = 1:T
        Hccc(:,:,t) = diag(h_vol(t,:))*R_ccc*diag(h_vol(t,:));
    end
  
  
    T = length(res(p+1:end,:)); H_true = Sigma(:,:,p+1:end);
    distdcc = zeros(T,1); distccc = zeros(T,1);
    dist_scad = zeros(T,1); dist_mcp = zeros(T,1);
    dist_npen = zeros(T,1);
    for tt = 1:T
        dist_scad(tt) = norm2(H_true(:,:,tt),H_scad(:,:,tt),N);
        dist_npen(tt) = norm2(H_true(:,:,tt),H_npen(:,:,tt),N);
        dist_mcp(tt) = norm2(H_true(:,:,tt),H_mcp(:,:,tt),N);
        distdcc(tt) = norm2(H_true(:,:,tt),Hdcc(:,:,tt),N);
        distccc(tt) = norm2(H_true(:,:,tt),Hccc(:,:,tt),N);
    end
    
    check(oo,:) = [mean(distdcc),mean(distccc),mean(dist_npen),mean(dist_scad),mean(dist_mcp)];
end


%% MARCH based DGP; N = 100
% the experiments for dimensions>=50 were launched on a cluster (Intel(R)-Xeon(R) with CPU E7-8891 v3, 2.80GHz, 1.2 Terad byte)
clear; clc;
Nsim = 100; check = zeros(Nsim,5);
N = 100; T = 800;
for oo = 1:Nsim
    
    [res,Sigma] = DGP_march100(T,N);
    
    len = T/2; folds = 4;
    
    p = 5; 
    lambda = 0.2*(0.01:0.01:0.5)*sqrt(log(p*N^2)/T);  
    [H_scad,b_scad,~,~,~,~] = SV_process_memo(res,p,'scad','no-constant',lambda,len,folds);
    
    [H_mcp,b_mcp,~,~,~,~] = SV_process_memo(res,p,'mcp','no-constant',lambda,len,folds);

    [H_npen,b_nonpen,~,~,~,~] = SV_process_memo(res,p,'nonpen','no-constant',lambda,len,folds);
    
    res_temp = res(p+1:end,:); T = length(res_temp);
    [parameters_dcc,Rt]=dcc_mvgarch(res_temp);
    h_vol=zeros(length(res_temp),size(res_temp,2));
    index = 1;
    for jj=1:size(res_temp,2)
        univariateparameters=parameters_dcc(index:index+1+1);
        [simulatedata, h_vol(:,jj)] = dcc_univariate_simulate(univariateparameters,1,1,res_temp(:,jj));
        index=index+1+1+1;
    end
    clear jj
    h_vol = sqrt(h_vol);
    [~,Rt_dcc,~,~]=dcc_mvgarch_full_likelihood(parameters_dcc,res_temp);
    Hdcc = zeros(N,N,length(res_temp));
    for t = 1:T
        Hdcc(:,:,t) = diag(h_vol(t,:))*Rt_dcc(:,:,t)*diag(h_vol(t,:));
    end
    
    Hccc = zeros(N,N,length(res_temp)); R_ccc = corr(res_temp);
    for t = 1:T
        Hccc(:,:,t) = diag(h_vol(t,:))*R_ccc*diag(h_vol(t,:));
    end

    
    T = length(res(p+1:end,:)); H_true = Sigma(:,:,p+1:end);
    distdcc = zeros(T,1); distccc = zeros(T,1); 
    dist_scad = zeros(T,1); dist_mcp = zeros(T,1);
    dist_npen = zeros(T,1);
    for tt = 1:T
        dist_scad(tt) = norm2(H_true(:,:,tt),H_scad(:,:,tt),N);
        dist_npen(tt) = norm2(H_true(:,:,tt),H_npen(:,:,tt),N);
        dist_mcp(tt) = norm2(H_true(:,:,tt),H_mcp(:,:,tt),N);
        distdcc(tt) = norm2(H_true(:,:,tt),Hdcc(:,:,tt),N);
        distccc(tt) = norm2(H_true(:,:,tt),Hccc(:,:,tt),N);
    end
    
    check(oo,:) = [mean(distdcc),mean(distccc),mean(dist_npen),mean(dist_scad),mean(dist_mcp)];
end


%% BEKK based DGP; N = 15
clear; clc;
Nsim = 100; check = zeros(Nsim,5);
N = 15; T = 800;

for oo = 1:Nsim
    
    [res,Sigma] = DGP_bekk15(T,N);
    
    len = T/2; folds = 4;
  
    p = 30;
    lambda = 0.5*(0.01:0.01:0.5)*sqrt(log(p*N^2)/T);  
    [H_scad,b_scad,~,~,~,~] = SV_process_memo(res,p,'scad','no-constant',lambda,len,folds);
    
    [H_mcp,b_mcp,~,~,~,~] = SV_process_memo(res,p,'mcp','no-constant',lambda,len,folds);
    
    [H_npen,b_nonpen,~,~,~,~] = SV_process_memo(res,p,'nonpen','no-constant',lambda,len,folds);
    
    res_temp = res(p+1:end,:); T = length(res_temp);
    [parameters_dcc,Rt]=dcc_mvgarch(res_temp);
    h_vol=zeros(length(res_temp),size(res_temp,2));
    index = 1;
    for jj=1:size(res_temp,2)
        univariateparameters=parameters_dcc(index:index+1+1);
        [simulatedata, h_vol(:,jj)] = dcc_univariate_simulate(univariateparameters,1,1,res_temp(:,jj));
        index=index+1+1+1;
    end
    clear jj
    h_vol = sqrt(h_vol);
    [~,Rt_dcc,~,~]=dcc_mvgarch_full_likelihood(parameters_dcc,res_temp);
    Hdcc = zeros(N,N,length(res_temp));
    for t = 1:T
        Hdcc(:,:,t) = diag(h_vol(t,:))*Rt_dcc(:,:,t)*diag(h_vol(t,:));
    end
    
    Hccc = zeros(N,N,length(res_temp)); R_ccc = corr(res_temp);
    for t = 1:T
        Hccc(:,:,t) = diag(h_vol(t,:))*R_ccc*diag(h_vol(t,:));
    end
    
    T = length(res(p+1:end,:)); H_true = Sigma(:,:,p+1:end);
    distdcc = zeros(T,1); distccc = zeros(T,1);
    dist_scad = zeros(T,1); dist_mcp = zeros(T,1);
    dist_npen = zeros(T,1);
    for tt = 1:T
        dist_scad(tt) = norm2(H_true(:,:,tt),H_scad(:,:,tt),N);
        dist_npen(tt) = norm2(H_true(:,:,tt),H_npen(:,:,tt),N);
        dist_mcp(tt) = norm2(H_true(:,:,tt),H_mcp(:,:,tt),N);
        distdcc(tt) = norm2(H_true(:,:,tt),Hdcc(:,:,tt),N);
        distccc(tt) = norm2(H_true(:,:,tt),Hccc(:,:,tt),N);
    end
    
    check(oo,:) = [mean(distdcc),mean(distccc),mean(dist_npen),mean(dist_scad),mean(dist_mcp)];
end
%% BEKK based DGP; N = 50
% the experiments for dimensions>=50 were launched on a cluster (Intel(R)-Xeon(R) with CPU E7-8891 v3, 2.80GHz, 1.2 Terad byte)
clear;clc;
Nsim = 100; check = zeros(Nsim,5);
N = 50; T = 800;

for oo = 1:Nsim
    
    [res,Sigma] = DGP_bekk50(T,N);
    
    len = T/2; folds = 4;
   
    p = 15;
    lambda = 0.5*(0.01:0.01:0.5)*sqrt(log(p*N^2)/T); 
    [H_scad,b_scad,~,~,~,~] = SV_process_memo(res,p,'scad','no-constant',lambda,len,folds);
    
    [H_mcp,b_mcp,~,~,~,~] = SV_process_memo(res,p,'mcp','no-constant',lambda,len,folds); 
      
    [H_npen,b_nonpen,~,~,~,~] = SV_process_memo(res,p,'nonpen','no-constant',lambda,len,folds);
    
    res_temp = res(p+1:end,:); T = length(res_temp);
    [parameters_dcc,Rt]=dcc_mvgarch(res_temp);
    h_vol=zeros(length(res_temp),size(res_temp,2));
    index = 1;
    for jj=1:size(res_temp,2)
        univariateparameters=parameters_dcc(index:index+1+1);
        [simulatedata, h_vol(:,jj)] = dcc_univariate_simulate(univariateparameters,1,1,res_temp(:,jj));
        index=index+1+1+1;
    end
    clear jj
    h_vol = sqrt(h_vol);
    [~,Rt_dcc,~,~]=dcc_mvgarch_full_likelihood(parameters_dcc,res_temp);
    Hdcc = zeros(N,N,length(res_temp));
    for t = 1:T
        Hdcc(:,:,t) = diag(h_vol(t,:))*Rt_dcc(:,:,t)*diag(h_vol(t,:));
    end
    
    Hccc = zeros(N,N,length(res_temp)); R_ccc = corr(res_temp);
    for t = 1:T
        Hccc(:,:,t) = diag(h_vol(t,:))*R_ccc*diag(h_vol(t,:));
    end
  
    T = length(res(p+1:end,:)); H_true = Sigma(:,:,p+1:end);
    distdcc = zeros(T,1); distccc = zeros(T,1); 
    dist_scad = zeros(T,1); dist_mcp = zeros(T,1);
    dist_npen = zeros(T,1);
    for tt = 1:T
        dist_scad(tt) = norm2(H_true(:,:,tt),H_scad(:,:,tt),N);
        dist_npen(tt) = norm2(H_true(:,:,tt),H_npen(:,:,tt),N);
        dist_mcp(tt) = norm2(H_true(:,:,tt),H_mcp(:,:,tt),N);
        distdcc(tt) = norm2(H_true(:,:,tt),Hdcc(:,:,tt),N);
        distccc(tt) = norm2(H_true(:,:,tt),Hccc(:,:,tt),N);
    end
    
    check(oo,:) = [mean(distdcc),mean(distccc),mean(dist_npen),mean(dist_scad),mean(dist_mcp)];
end
%% BEKK based DGP; N = 100
% the experiments for dimensions>=50 were launched on a cluster (Intel(R)-Xeon(R) with CPU E7-8891 v3, 2.80GHz, 1.2 Terad byte)
clear; clc;
Nsim = 100; check = zeros(Nsim,5);
N = 100; T = 800;

for oo = 1:Nsim

    
    
    [res,Sigma] = DGP_bekk100(T,N);
    
    len = T/2; folds = 4;
    
    p = 5;
    lambda = 0.5*(0.01:0.01:0.5)*sqrt(log(p*N^2)/T); 
    [H_scad,b_scad,~,~,~,~] = SV_process_memo(res,p,'scad','no-constant',lambda,eta_p,len,3);
       
    [H_mcp,b_mcp,~,~,~,~] = SV_process_memo(res,p,'mcp','no-constant',lambda,eta_p,len,3);
    
    [H_npen,b_nonpen,~,~,~,~] = SV_process_memo(res,p,'nonpen','no-constant',lambda,eta_p,len,3);
    
    res_temp = res(p+1:end,:); T = length(res_temp);
    [parameters_dcc,Rt]=dcc_mvgarch(res_temp);
    h_vol=zeros(length(res_temp),size(res_temp,2));
    index = 1;
    for jj=1:size(res_temp,2)
        univariateparameters=parameters_dcc(index:index+1+1);
        [simulatedata, h_vol(:,jj)] = dcc_univariate_simulate(univariateparameters,1,1,res_temp(:,jj));
        index=index+1+1+1;
    end
    clear jj
    h_vol = sqrt(h_vol);
    [~,Rt_dcc,~,~]=dcc_mvgarch_full_likelihood(parameters_dcc,res_temp);
    Hdcc = zeros(N,N,length(res_temp));
    for t = 1:T
        Hdcc(:,:,t) = diag(h_vol(t,:))*Rt_dcc(:,:,t)*diag(h_vol(t,:));
    end
    
    
    Hccc = zeros(N,N,length(res_temp)); R_ccc = corr(res_temp);
    for t = 1:T
        Hccc(:,:,t) = diag(h_vol(t,:))*R_ccc*diag(h_vol(t,:));
    end

    T = length(res(p+1:end,:)); H_true = Sigma(:,:,p+1:end);
    distdcc = zeros(T,1); distccc = zeros(T,1);
    dist_scad = zeros(T,1); dist_mcp = zeros(T,1);
    dist_npen = zeros(T,1);
    for tt = 1:T
        dist_scad(tt) = norm2(H_true(:,:,tt),H_scad(:,:,tt),N);
        dist_npen(tt) = norm2(H_true(:,:,tt),H_npen(:,:,tt),N);
        dist_mcp(tt) = norm2(H_true(:,:,tt),H_mcp(:,:,tt),N);
        distdcc(tt) = norm2(H_true(:,:,tt),Hdcc(:,:,tt),N);
        distccc(tt) = norm2(H_true(:,:,tt),Hccc(:,:,tt),N);
    end
    
    check(oo,:) = [mean(distdcc),mean(distccc),mean(dist_npen),mean(dist_scad),mean(dist_mcp)];

end

