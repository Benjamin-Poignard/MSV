function theta = ols_regularized(X,Y,lambda,a_scad,b_mcp,method)

[n,d] = size(X);
if d > n
    start = zeros(d,1); % From the null model, if p > n
else
    start = X \ Y;  % From the OLS estimate, if p <= n
end

param_update = start; param = param_update;

ii = 0; maxIt = 1e4; tol = 1e-10;
while ii < maxIt
    
    %Z = -X'*(Y-X*param_update)/n;
    Z = (X'*X)\(X'*Y);
    param_est = zeros(d,1);
    switch method
        case 'scad'
            for ii = 1:d
                if (abs(Z(ii)) <= 2*lambda)
                    param_est(ii) = soft_thresholding(Z(ii),lambda);
                elseif (2*lambda < abs(Z(ii)) && abs(Z(ii)) <= a_scad*lambda)
                    param_est(ii) = ((a_scad-1)/(a_scad-2))*soft_thresholding(Z(ii),(a_scad*lambda)/(a_scad-1));
                elseif (a_scad*lambda < abs(Z(ii)))
                    param_est(ii) = Z(ii);
                end
            end
            param_update = param_est;
        case 'mcp'
            for ii = 1:d
                if (abs(Z(ii)) <= b_mcp*lambda)
                    param_est(ii) = (b_mcp/(b_mcp-1))*soft_thresholding(Z(ii),lambda);
                elseif (b_mcp*lambda < abs(Z(ii)))
                    param_est(ii) = Z(ii);
                end
            end
            param_update = param_est;
    end
    delta = norm(param_update-param,2);
    if delta < tol, break; end
    param = param_update;
    
end
param_update(abs(param_update)<0.00001)=0;
theta = param_update;
