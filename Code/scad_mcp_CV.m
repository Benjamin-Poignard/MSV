function [b_L,lambda_opt] = scad_mcp_CV(Y,XX,scad,mcp,lambda,K,method)

[n,p] = size(XX); len = round(n/K);
XX_f = zeros(len,p,K); y_f = zeros(len,K);
theta_fold = zeros(p,length(lambda),K);
hv = round(0.05*n);
y_f(:,1) = Y(1:len,:); y_temp = Y; y_temp(1:len+hv,:) = [];
XX_f(:,:,1) = XX(1:len,:); XX_temp = XX; XX_temp(1:len+hv,:)=[];
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
theta_fold(:,:,1) = B_up;
for kk = 2:K-1
    y_f(:,kk) = Y((kk-1)*len+1:kk*len,:); y_temp = Y; y_temp((kk-1)*len+1-hv:kk*len+hv,:) = [];
    XX_f(:,:,kk) = XX((kk-1)*len+1:kk*len,:); XX_temp = XX; XX_temp((kk-1)*len+1-hv:kk*len+hv,:) = [];
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

y_f(:,K) = Y(end-len+1:end,:); y_temp = Y; y_temp(end-len+1-hv:end,:) = [];
XX_f(:,:,K) = XX(end-len+1:end,:); XX_temp = XX; XX_temp(end-len+1-hv:end,:) = [];
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
theta_fold(:,:,K) = B_up;

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