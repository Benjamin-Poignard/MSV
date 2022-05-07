function [b_adaL,lambda_opt,gamma_opt] = adaptive_Lasso_CV(Y,XX,lambda,gamma,len,K)

[n,p] = size(XX);
XX_f = zeros(len,p,K); y_f = zeros(len,K);
theta_fold = zeros(length(lambda)*p,length(gamma),K);

for kk = 1:K
    
    cond = true;
    while cond
        choose = round(((n-len)*rand(1))+1);
        cond = (choose > n-len-5);
    end
    
    y_temp = Y; XX_temp = XX;
    y_temp(choose+1:choose+len) = []; XX_temp(choose+1:choose+len,:) = [];
    
    y_f(:,kk) = Y(choose+1:choose+len); XX_f(:,:,kk) = XX(choose+1:choose+len,:);
    
    b_g = [];
    for ii = 1:length(lambda)
        b_temp = [];
        for jj = 1:length(gamma)
            b_up = lassoShooting(XX_temp,y_temp,lambda(ii),gamma(jj));
            b_temp = [b_temp b_up];
        end
        b_g = [b_g ; b_temp];
    end
    
    theta_fold(:,:,kk) = b_g;
    % clear x_fold y_fold
    
end

L_lam = length(lambda); theta_up = {};
for kk = 1:K
    for ll = 1:L_lam
        theta_up{kk}(:,:,ll) = theta_fold(1+(ll-1)*p:ll*p,:,kk);
    end
end
clear kk

count = zeros(length(lambda),length(gamma));
for ii = 1:length(lambda)
    for jj = 1:length(gamma)
        for kk = 1:K
            count(ii,jj) = count(ii,jj) + sum((y_f(:,kk)-XX_f(:,:,kk)*theta_up{kk}(:,jj,ii)).^2);
        end
    end
end
clear ii jj
[ii,jj] = find(count==min(min(count)));
lambda_opt = lambda(ii); gamma_opt = gamma(jj);
b_adaL = lassoShooting(XX,Y,lambda_opt,gamma_opt);
