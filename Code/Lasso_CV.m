function [b_L,lambda_opt]= Lasso_CV(Y,XX,lambda,len,K)

[n,p] = size(XX);
XX_f = zeros(len,p,K); y_f = zeros(len,K);
theta_fold = zeros(p,length(lambda),K);

for kk = 1:K
    hv = 10;
    cond = true;
    while cond
        choose = round(((n-len)*rand(1))+1);
        cond = (choose+len+hv>n)||(choose<hv);
    end

    y_temp = Y; XX_temp = XX;
    y_temp(choose+1-hv:choose+len+hv) = []; XX_temp(choose+1-hv:choose+len+hv,:) = [];
    y_f(:,kk) = Y(choose+1:choose+len); XX_f(:,:,kk) = XX(choose+1:choose+len,:);
    
    B_up = [];
    for nn = 1:length(lambda)
        b_up = lassoShooting(XX_temp,y_temp,lambda(nn),0);
        B_up = [B_up,b_up];
    end
    theta_fold(:,:,kk) = B_up;
    
end
count = zeros(length(lambda),1);
for ii = 1:length(lambda)
   for kk = 1:K
       count(ii) = count(ii) + sum((y_f(:,kk)-XX_f(:,:,kk)*theta_fold(:,ii,kk)).^2);
   end
end
clear ii
[~,ind] = min(count); lambda_opt = lambda(ind);
b_L = lassoShooting(XX,Y,lambda_opt,0); b_L(abs(b_L)<0.0001)=0;
clear ind
