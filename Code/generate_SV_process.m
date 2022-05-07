function H = generate_SV_process(data,p,B_hat,Sig_zeta,Sig_alpha,Gamma)

% This function generates the variance covariance MSV process H from the
% parameters estimated with SV_process_estim_memo.m
% H can be both the out-of-sample variance covariance when data are the
% out-of-sample observations, or the in-sample variance covariance

% - data: T x N vector of observations
% - p: number of lags for the first step 
% - B_hat, Sig_zeta, Sig_alpha and Gamma are all the model parameters
% obtained with SV_process_estim_memo.m

% output: H, which is a N x N x T variance covariance process


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

c_star = B_hat(:,1); B_hat(:,1) = [];
Phi_sec = B_hat(:,1:N); B_hat(:,1:N) = [];

% Compute the following quantities
c_hat = inv(eye(N)-Phi_sec)*c_star;

%%%%%%%%%%%%%%%%%%%%%% Fourth step: MMSLE %%%%%%%%%%%%%%%%%%%%%

c_vec = kron(ones(T,1),c_hat); % size of the vector: TN

% computation of the V_alpha matrix
V_alpha = zeros(T*N);
for ii = 1:T
    for jj = ii:T
        V_alpha(1+(ii-1)*N:ii*N,1+(jj-1)*N:jj*N) = Sig_alpha*(Phi_sec')^(jj-ii);
        V_alpha(1+(jj-1)*N:jj*N,1+(ii-1)*N:ii*N) = V_alpha(1+(ii-1)*N:ii*N,1+(jj-1)*N:jj*N)';
    end
end
V_zeta = kron(eye(T),Sig_zeta);
V_alpha = proj_defpos(V_alpha);
% Obtain V_x
V_x = V_alpha+V_zeta;

% Compute x_tilde
x_tilde = V_alpha*inv(V_x)*(vec(x)-c_vec)+c_vec;

% Generate the \tilde{D}_t matrix
xx_tilde = reshape(x_tilde,N,T);

d_bar = zeros(N,1); d_tilde = zeros(T,N);
for ii = 1:N
    d_bar(ii) = sqrt(sum((data(:,ii).^2).*exp(-xx_tilde(ii,:)'))/(T-p));
    d_tilde(:,ii) = d_bar(ii)*exp(0.5*xx_tilde(ii,:)');
end

% Generate the variance covariance matrix H using \tilde{D}_t and Gamma
H = zeros(N,N,T);
for t = 1:T
    H(:,:,t) = diag(d_tilde(t,:))*Gamma*diag(d_tilde(t,:));
end
