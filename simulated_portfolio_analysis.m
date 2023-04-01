% Simulated portfolio
% The estimation for the MSV models were performed on a cluster:
% Intel(R)-Xeon(R) with CPU E7-8891 v3, 2.80GHz, 1.2 Terad byte

% Note: the following section specifies the data generating process (DGP)
% that simulates a vector "res" based on the decomposition
% res = mvnrnd(0,H_t), that is a multivariate normal distribution, centered
% with dynamic variance covariance H_t
% H_t is deduced from the decomposition H_t = Q_t C_t Q_t, with Q_t
% diagonal containing GARCH(1,1) volatilities; C_t is a correlation matrix
% deduced from cos/sin/constant with jumps dynamics
% This DGP favors the DCC/GOGARCH estimation since the decomposition of H_t
% and the marginals in Q_t are the same as the DCC

clear; clc;
% N is the dimension of the vector of observations
% T is the full sample size
N = 5; T = 5001;
hsim2 = zeros(T,N); hsim2(1,:) = 0.005.*ones(1,N); res = zeros(T,N);
hsim = zeros(T,N); hsim(1,:) = sqrt(hsim2(1,:));
Sigma = zeros(N,N,T);
Correlation = zeros(N,N,T);
constant = 0.0001 + (0.009-0.0001)*rand(1,N);
gamma = true;
while(gamma)
    b = 0.7 + (0.95-0.7)*rand(1,N);
    a = 0.01 + (0.15-0.01)*rand(1,N);
    gamma = (any(b+a > 1));
end

normalisation = [50;100;200;500;1200;1500];
aaa = 1+round(rand(N*(N-1)/2,1)*5);
bbb = 1+round(rand(N*(N-1)/2,1)*3); gg = {'cos','sin','const','mode'};
coefficient = -0.3+0.6*rand(N*(N-1)/2,2); d_const = randi([1 T],1);

L = zeros(N*(N-1)/2,T);
const = -0.5+1*rand(1,1); gamma = 3;

for t = 2:T
    
    hsim2(t,:) = constant + b.*hsim2(t-1,:) + a.*(res(t-1,:).^2);
    hsim(t,:) = sqrt(hsim2(t,:));
    for ii = 1:N*(N-1)/2
        option = char(gg(bbb(ii)));
        switch option
            case 'cos'
                pp = cos(2*pi*t/normalisation(aaa(ii)));
            case 'sin'
                pp = sin(2*pi*t/normalisation(aaa(ii)));
            case 'mode'
                pp = mode(t/normalisation(aaa(ii)),1);
            case 'const'
                pp = double(t>d_const);
        end
        L(ii,t) = coefficient(ii,1)+coefficient(ii,2)*pp;
    end
    clear ii
    
    C = tril(vech_off(L(:,t),N)); Ctemp = C*C';
    Correlation(:,:,t) = Ctemp./(sqrt(diag(Ctemp))*sqrt(diag(Ctemp))');
    Sigma(:,:,t) = diag(hsim(t,:))*Correlation(:,:,t)*diag(hsim(t,:));
    % alternatively, a student distribution can be specified
    res(t,:) = mvnrnd(zeros(N,1),Sigma(:,:,t));
    
end
clear t
res = res(2:end,:);

% Set the in-sample estimation period: up to T_in
% Set the out-of-sample period for DM/MCS tests: from date T_out
T_in = 4300; T_out = T_in+1;
% The variance models are estimated using res_in
% The DM/MCS tests are performed using res_out
res_in = res(1:T_in,:); res_out = res(T_out:end,:);

%% Alternative DGP:
% one can simulate a Cholesky type dynamic for H_t, that is
% H_t = B_tB'_t, with B_t lower triangular with dynamic coefficients

%% Alternative DGP: MARCH type dynamic as in the simulations
clear
clc
N = 15; T = 5001;
[res,Sigma] = DGP_march15(T,N);

T_in = 4300; T_out = T_in+1;
res_in = res(1:T_in,:); res_out = res(T_out:end,:);


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% Estimation of (non-)penalized MSV based processes %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

folds = 5;

% A cross-validation procedure is performed to select the optimal tuning
% parameter lambda. A grid search is specified around sqrt(log(p*N^2)/T),
% which is a rate generally identified as optimal in the sparse literature
% One can specify a wider/smaller grid

% estimate the MSV model with p=10 lags
p = 10;
grid = (0.01:0.2:3); lambda = grid*sqrt(log(p*N^2)/T);

% Adaptive lasso penalized MSV
[~,B,Sig_zeta,Sig_alpha,Gamma] = SV_process_estimate(res_in,p,'alasso','no-constant',lambda,folds);
% Generate the out-of-sample forecasts of the Adaptive Lasso MSV using res_out
H_msv_ols_al = generate_SV_process(res_out,p,B,Sig_zeta,Sig_alpha,Gamma);

% SCAD penalized MSV
[~,B,Sig_zeta,Sig_alpha,Gamma] = SV_process_estimate(res_in,p,'scad','no-constant',lambda,folds);
% Generate the out-of-sample forecasts of the SCAD MSV using res_out
H_msv_ols_scad = generate_SV_process(res_out,p,B,Sig_zeta,Sig_alpha,Gamma);

% MCP penalized MSV
[~,B,Sig_zeta,Sig_alpha,Gamma] = SV_process_estimate(res_in,p,'mcp','no-constant',lambda,folds);
% Generate the out-of-sample forecasts of the MCP MSV using res_out
H_msv_ols_mcp = generate_SV_process(res_out,p,B,Sig_zeta,Sig_alpha,Gamma);

% Non-penalized MSV
[~,B,Sig_zeta,Sig_alpha,Gamma] = SV_process_estimate(res_in,p,'nonpen','no-constant',lambda,folds);
% Generate the out-of-sample forecasts of the Non-penalised MSV using res_out
H_msv_ols_nonpen = generate_SV_process(res_out,p,B,Sig_zeta,Sig_alpha,Gamma);


% estimate the MSV model with p=20 lags
p = 20;
grid = (0.01:0.2:3); lambda = grid*sqrt(log(p*N^2)/T);

% Adaptive lasso penalised MSV
[~,B,Sig_zeta,Sig_alpha,Gamma] = SV_process_estimate(res_in,p,'alasso','no-constant',lambda,folds);
% Generate the out-of-sample forecasts of the Adaptive Lasso MSV using res_out
H_msv_ols_al_l = generate_SV_process(res_out,p,B,Sig_zeta,Sig_alpha,Gamma);

% SCAD penalized MSV
[~,B,Sig_zeta,Sig_alpha,Gamma] = SV_process_estimate(res_in,p,'scad','no-constant',lambda,folds);
% Generate the out-of-sample forecasts of the SCAD MSV using res_out
H_msv_ols_scad_l = generate_SV_process(res_out,p,B,Sig_zeta,Sig_alpha,Gamma);

% MCP penalized MSV
[~,B,Sig_zeta,Sig_alpha,Gamma] = SV_process_estimate(res_in,p,'mcp','no-constant',lambda,folds);
% Generate the out-of-sample forecasts of the MCP MSV using res_out
H_msv_ols_mcp_l = generate_SV_process(res_out,p,B,Sig_zeta,Sig_alpha,Gamma);

% Non-penalized MSV
lambda = 0;
[~,B,Sig_zeta,Sig_alpha,Gamma] = SV_process_estimate(res_in,p,'nonpen','no-constant',lambda,folds);
% Generate the out-of-sample forecasts of the Non-penalised MSV using res_out
H_msv_ols_nonpen_l = generate_SV_process(res_out,p,B,Sig_zeta,Sig_alpha,Gamma);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%% Estimation of scalar DCC model %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Note that all the MGARCH models are estimated/generated from the functions
% that where downloaded from the MFE toolbox: please see
% https://www.kevinsheppard.com/code/matlab/mfe-toolbox/

% scalar DCC
[parameters_dcc, Rt,H_in]=dcc_mvgarch(res_in);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Generation of the out-of-sample forecasts for the scalar DCC %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h_oos=zeros(length(res_out),size(res_out,2)); index = 1;
for jj=1:size(res_out,2)
    univariateparameters=parameters_dcc(index:index+1+1);
    [simulatedata, h_oos(:,jj)] = dcc_univariate_simulate(univariateparameters,1,1,res_out(:,jj));
    index=index+1+1+1;
end
clear jj
h_oos = sqrt(h_oos); [~,Rt_dcc_oos,~,~]=dcc_mvgarch_generate_oos(parameters_dcc,res_out,res_in,H_in);

Hdcc = zeros(N,N,length(res_out)); T = length(res_out);
for t = 1:T
    Hdcc(:,:,t) = diag(h_oos(t,:))*Rt_dcc_oos(:,:,t)*diag(h_oos(t,:));
end

% DM-test: first, obtain the GMVP based portfolio weights
wdcc = zeros(N,T);
wmsv_unpen = zeros(N,T); wmsv_al = zeros(N,T); wmsv_scad = zeros(N,T); wmsv_mcp = zeros(N,T);
wmsv_unpen_l = zeros(N,T); wmsv_al_l = zeros(N,T); wmsv_scad_l = zeros(N,T); wmsv_mcp_l = zeros(N,T);


for t = 1:T
    wdcc(:,t)= GMVP(Hdcc(:,:,t));
    wmsv_unpen(:,t)= GMVP(H_msv_ols_nonpen(:,:,t));
    wmsv_al(:,t)= GMVP(H_msv_ols_al(:,:,t));
    wmsv_scad(:,t)= GMVP(H_msv_ols_scad(:,:,t));
    wmsv_mcp(:,t)= GMVP(H_msv_ols_mcp(:,:,t));
    wmsv_unpen_l(:,t)= GMVP(H_msv_ols_nonpen_l(:,:,t));
    wmsv_al_l(:,t)= GMVP(H_msv_ols_al_l(:,:,t));
    wmsv_scad_l(:,t)= GMVP(H_msv_ols_scad_l(:,:,t));
    wmsv_mcp_l(:,t)= GMVP(H_msv_ols_mcp_l(:,:,t));
end

e1 = zeros(T,1); e2 = zeros(T,1); e3 = zeros(T,1); e4 = zeros(T,1);
e5 = zeros(T,1); e6 = zeros(T,1); e7 = zeros(T,1); e8 = zeros(T,1);
e9 = zeros(T,1);


for t = 1:T
    e1(t) = wdcc(:,t)'*res_out(t,:)';
    e2(t) = wmsv_unpen(:,t)'*res_out(t,:)';
    e3(t) = wmsv_al(:,t)'*res_out(t,:)';
    e4(t) = wmsv_scad(:,t)'*res_out(t,:)';
    e5(t) = wmsv_mcp(:,t)'*res_out(t,:)';
    e6(t) = wmsv_unpen_l(:,t)'*res_out(t,:)';
    e7(t) = wmsv_al_l(:,t)'*res_out(t,:)';
    e8(t) = wmsv_scad_l(:,t)'*res_out(t,:)';
    e9(t) = wmsv_mcp_l(:,t)'*res_out(t,:)';
end
e1 = e1.^2; e2 = e2.^2; e3 = e3.^2; e4 = e4.^2; e5 = e5.^2; e6 = e6.^2; e7 = e7.^2;
e8 = e8.^2; e9 = e9.^2;
E = [e1 e2 e3 e4 e5 e6 e7 e8 e9];

% Generate the table for the DM test results, where the columns are as:
% DCC & MSV(10) & MSV-AL(10) & MSV_SCA(10) & MSV_MCP(10) & MSV(20) & MSV-AL(20) & MSV_SCA(20) & MSV_MCP(20)
DM2 = [];
for kk = 1:size(E,2)
    DM = [];
    for ii = 1:size(E,2)
        DM = [ DM , dmtest(E(:,kk),E(:,ii),1) ];
    end
    DM2 = [DM2;DM];
end

% Model Confidence Test
[includedR, pvalsR, excluded] = mcs(E,0.05,1000,12);
[[excluded ;includedR] pvalsR]
