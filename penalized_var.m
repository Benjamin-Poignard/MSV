function [b_est,lambda_opt] = penalized_var(Y,X,lambda,method,K)

% Inputs:
%         - Y: vector of response variables
%         - X: matrix of lagged variables
%         - lambda: vector of candidates for the tuning parameter (user
%           specified)
%         - method: 'lasso', 'alasso', 'scad' or 'mcp'
%           'alasso' stands for adaptive LASSO
%           'lasso' and 'alasso' are solved by the Shooting algorithm
%           'scad' and 'mcp' are numerically solved by fmincon
%         - K: number of folds (should be strictly larger than 2)
% Outputs:
%         - b_est: vector of estimated parameters
%         - lambda_opt: optimal tuning parameter selected by
%           cross-validation for the corresponding penalty function

[n,p] = size(X); len = round(n/K);
X_f = zeros(len,p,K); y_f = zeros(len,K);
theta_fold = zeros(p,length(lambda),K);
hv = round(0.05*n);
y_f(:,1) = Y(1:len,:); y_temp = Y; y_temp(1:len+hv,:) = [];
X_f(:,:,1) = X(1:len,:); X_temp = X; X_temp(1:len+hv,:)=[];
B_up = [];
parfor nn = 1:length(lambda)
    b_up = VAR_penalized(y_temp,X_temp,lambda(nn),method);
    B_up = [B_up b_up];
end
theta_fold(:,:,1) = B_up;
for kk = 2:K-1
    y_f(:,kk) = Y((kk-1)*len+1:kk*len,:); y_temp = Y; y_temp((kk-1)*len+1-hv:kk*len+hv,:) = [];
    X_f(:,:,kk) = X((kk-1)*len+1:kk*len,:); X_temp = X; X_temp((kk-1)*len+1-hv:kk*len+hv,:) = [];
    B_up = [];
    parfor nn = 1:length(lambda)
        b_up = VAR_penalized(y_temp,X_temp,lambda(nn),method);
        B_up = [B_up b_up];
    end
    theta_fold(:,:,kk) = B_up;
end

y_f(:,K) = Y(end-len+1:end,:); y_temp = Y; y_temp(end-len+1-hv:end,:) = [];
X_f(:,:,K) = X(end-len+1:end,:); X_temp = X; X_temp(end-len+1-hv:end,:) = [];
B_up = [];
parfor nn = 1:length(lambda)
    b_up = VAR_penalized(y_temp,X_temp,lambda(nn),method);
    B_up = [B_up b_up];
end
theta_fold(:,:,K) = B_up;

count = zeros(length(lambda),1);
for ii = 1:length(lambda)
    for kk = 1:K
        count(ii) = count(ii) + sum((y_f(:,kk)-X_f(:,:,kk)*theta_fold(:,ii,kk)).^2);
    end
    count(ii) = count(ii)/K;
end
clear ii
[~,ind] = min(count); lambda_opt = lambda(ind);
b_est = VAR_penalized(Y,X,lambda_opt,method);
