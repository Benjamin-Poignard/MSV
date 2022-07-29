function [b_L,lambda_opt] = scad_mcp_CV(Y,XX,scad,mcp,lambda,len,K,method)

[n,p] = size(XX);
XX_f = zeros(len,p,K); y_f = zeros(len,K);
theta_fold = zeros(p,length(lambda),K);

for kk = 1:K
    hv = round(0.05*n); choose = round(((n-len)*rand(1))+1);
    if and(choose+hv+len<n,choose>hv)
        y_f(:,kk) = Y(choose+1:choose+len,:);
        XX_f(:,:,kk) = XX(choose+1:choose+len,:);
        XX_temp = XX; XX_temp(choose+1-hv:choose+len+hv,:) = [];
        y_temp = Y; y_temp(choose+1-hv:choose+len+hv,:) = [];
    elseif (choose<hv)
        y_f(:,kk) = Y(1:len,:);
        XX_f(:,:,kk) = XX(1:len,:);
        XX_temp = XX; XX_temp(1:len+hv,:) = [];
        y_temp = Y; y_temp(1:len+hv,:) = [];
    elseif (choose+len+hv>n)
        y_f(:,kk) = Y(end-len+1:end,:);
        XX_f(:,:,kk) = XX(end-len+1:end,:);
        XX_temp = XX; XX_temp(end-len+1-hv:end,:) = [];
        y_temp = Y; y_temp(end-len+1-hv:end,:) = [];
    end
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