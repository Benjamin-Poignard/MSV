function [b_adaL,lambda_opt,gamma_opt] = adaptive_Lasso_CV(Y,XX,lambda,gamma,K)

[n,p] = size(XX); len = round(n/K);
XX_f = zeros(len,p,K); y_f = zeros(len,K);
theta_fold = zeros(length(lambda)*p,length(gamma),K);

hv = round(0.05*n);

y_f(:,1) = Y(1:len,:); y_temp = Y; y_temp(1:len+hv,:) = [];
XX_f(:,:,1) = XX(1:len,:); XX_temp = XX; XX_temp(1:len+hv,:)=[];
b_g = [];
for ii = 1:length(lambda)
    b_temp = [];
    for jj = 1:length(gamma)
        b_up = lassoShooting(XX_temp,y_temp,lambda(ii),gamma(jj));
        b_temp = [b_temp b_up];
    end
    b_g = [b_g ; b_temp];
end
theta_fold(:,:,1) = b_g;


for kk = 2:K-1
    y_f(:,kk) = Y((kk-1)*len+1:kk*len,:); y_temp = Y; y_temp((kk-1)*len+1-hv:kk*len+hv,:) = [];
    XX_f(:,:,kk) = XX((kk-1)*len+1:kk*len,:); XX_temp = XX; XX_temp((kk-1)*len+1-hv:kk*len+hv,:) = [];
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
end

y_f(:,K) = Y(end-len+1:end,:); y_temp = Y; y_temp(end-len+1-hv:end,:) = [];
XX_f(:,:,K) = XX(end-len+1:end,:); XX_temp = XX; XX_temp(end-len+1-hv:end,:) = [];
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
