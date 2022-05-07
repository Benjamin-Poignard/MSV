function [b_L,lambda_opt] = scad_mcp_CV(Y,XX,scad,mcp,lambda,len,K,method)

[n,p] = size(XX);
XX_f = zeros(len,p,K); y_f = zeros(len,K);
theta_fold = zeros(p,length(lambda),K);

for kk = 1:K
    cond = true;
    while cond
        choose = round(((n-len)*rand(1))+1);
        cond = (choose > n-len-5);
    end

    y_temp = Y; XX_temp = XX;
    y_temp(choose+1:choose+len) = []; XX_temp(choose+1:choose+len,:) = [];
    y_f(:,kk) = Y(choose+1:choose+len); XX_f(:,:,kk) = XX(choose+1:choose+len,:);
    
    B_up = [];
    for nn = 1:length(lambda)
        switch method
            case 'mcp'
                b_up = ols_regularized(XX_temp,y_temp,lambda(nn),scad,mcp,'mcp');
                B_up = [B_up b_up];
            case 'scad'
                b_up = ols_regularized(XX_temp,y_temp,lambda(nn),scad,mcp,'scad');
                B_up = [B_up b_up];
        end
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

switch method
    case 'mcp'
        b_L = ols_regularized(XX,Y,lambda_opt,scad,mcp,'mcp');
    case 'scad'
        b_L = ols_regularized(XX,Y,lambda_opt,scad,mcp,'scad');
end