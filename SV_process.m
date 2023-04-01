function [H,b,B_hat_save,Sig_zeta,Sig_alpha,Gamma] = SV_process(data,p,method,constant,lambda,K)

% Inputs:
%        - data: T x N vector of observations
%        - p: number of lags for the first step
%        - method: penalisation method ==> 'lasso', 'alasso', 'scad', 'mcp'
%        with alasso for adaptive lasso
%        - constant
%         ==> integrate intercept parameter in the first step: 'constant'
%         ==> no intercept parameter in the first step: 'no-constant'
%        - lambda: penalisation parameter
%        - K: number of folds for cross-validation

% Outputs:
%        - - H: N x N x T variance covariance process (in-sample)
%        - b: first step estimator
%        - B_hat: second step estimator
%        - Sig_zeta and Sig_alpha: please refers to the paper for the 
%          definitions of these quantities, which correspond to 
%          \Sigma_\zeta and \Sigma_\alpha
%        - Gamma: correlation estimator obtained in the third step

% T: number of simulated observations; N: dimension of the vector
[T,N] = size(data);

%%%%%% define the vectors and matrix %%%%%%
% Sigma is the true variance covariance matrix
% x corresponds to the log(data^2)

x = log(data.^2)';
%%%%%%%%%%%%%%%%%%%%%% First step: penalisation %%%%%%%%%%%%%%%%%%%%%%

%%% Equation by equation penalisation: this will be useful when considering
%%% large N
%%% Penalisation is performed for the lasso, adaptive lasso, scad, mcp and
%%% Bridge

% creation of the vector of covariate
X = [];
for tt = p+1:T
    x_temp_reg = [];
    for kk = 1:p
        x_temp_reg = [x_temp_reg ; x(:,tt-kk)];
    end
    X = [X , x_temp_reg];
end
YY = x(:,p+1:end); y = x'; b = [];

switch constant
    case 'constant'
        XX = [ones(T-p,1),X'];
    case 'no-constant'
        XX = X';
end

% equation-by-equation penalized estimation procedure
for ii = 1:N
    switch method
        case 'lasso'
            % lasso penalization
            [b_lasso,~]= penalized_var(y(p+1:end,ii),XX,lambda,'lasso',K);
            b = [b;b_lasso'];       
        case 'alasso'
            % adaptive lasso penalization
            [b_alasso,~] = penalized_var(y(p+1:end,ii),XX,lambda,'alasso',K);
            b = [b;b_alasso'];
        case 'scad'
            % scad penalization 
            [b_scad,~] = penalized_var(y(p+1:end,ii),XX,lambda,'scad',K);
            b = [b;b_scad'];
        case 'mcp'
            % mcp penalization 
            [b_mcp,~] = penalized_var(y(p+1:end,ii),XX,lambda,'mcp',K);
            b = [b;b_mcp'];
        case 'nonpen'
            % non-penalized model: simple OLS
            b_np = ((XX'*XX)\(XX'*y(p+1:end,ii)))';
            b = [b;b_np];
    end
end

% B corresponds to the estimated sparse Psi matrix in step 1
B = b; 
% obtain the residuals and variance covariance
u = YY-B*X; T = length(u); 

%%%%%%%%%%%%%%%%%%%%%% Second step: OLS estimation %%%%%%%%%%%%%%%%%%%%%%

% creation of the vector of covariates
x_second = x(:,p+1:end);
XX = [];
for tt = 2:T
    XX = [XX  [1;x_second(:,tt-1);u(:,tt-1)]];
end
% OLS estimator second step
Y_second = x_second(:,2:end); % dependent variable
B_hat = Y_second*XX'*inv(XX*XX'); B_hat_save = B_hat;
c_star = B_hat(:,1); B_hat(:,1) = [];
Phi_sec = B_hat(:,1:N); B_hat(:,1:N) = [];
Xi_sec = B_hat(:,1:N);


% Compute the following quantities
c_hat = inv(eye(N)-Phi_sec)*c_star;

%%%%%%%%%%%%%%%%%%%%%% Third step: correlation matrix %%%%%%%%%%%%%%%%%%%%%

Gamma = corr(data);%Gamma


%%%%%%%%%%%%%%%%%%%%%% Fourth step: MMSLE %%%%%%%%%%%%%%%%%%%%%

c_vec = kron(ones(T,1),c_hat); % size of the vector: TN

Sig_x = cov(x(:,p+1:end)');
rsb = 0.5*(pi^2)/mean(diag(Sig_x));
Sig_alpha = (1-rsb)*Sig_x;
Sig_zeta = rsb*Sig_x;

V_alpha = fV(Sig_alpha,Phi_sec,T);
V_zeta = kron(eye(T),Sig_zeta);
% Obtain V_x
V_x = V_alpha+V_zeta;

% Compute x_tilde
x_tilde = V_alpha*inv(V_x)*(vec(x(:,p+1:end))-c_vec)+c_vec;

% Generate the \tilde{D}_t matrix
xx_tilde = reshape(x_tilde,N,T);

d_bar = zeros(N,1); d_tilde = zeros(T,N);
for ii = 1:N
    d_bar(ii) = sqrt(sum((data(p+1:end,ii).^2).*exp(-xx_tilde(ii,:)'))/T);
    d_tilde(:,ii) = d_bar(ii)*exp(0.5*xx_tilde(ii,:)');
end

% Generate the variance covariance matrix H using \tilde{D}_t and Gamma
H = zeros(N,N,T);
for t = 1:T
    H(:,:,t) = diag(d_tilde(t,:))*Gamma*diag(d_tilde(t,:));
end