function penalty = mcp(param,lambda,b)

penalty = 0;
for ii = 1:length(param)
    if (abs(param(ii))<=b*lambda)
        shrinkage = lambda*abs(param(ii)) - (param(ii)^2)/(2*b);
    elseif (abs(param(ii))>b*lambda)
        shrinkage = b*lambda^2/2;
    end
    penalty = penalty + shrinkage;
end