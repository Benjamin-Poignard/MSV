function penalty = scad(param,lambda,a)

penalty = 0;
for ii = 1:length(param)
    if (abs(param(ii))<=lambda)
        shrinkage = lambda*abs(param(ii));
    elseif (lambda<abs(param(ii)) && abs(param(ii)) <=a*lambda)
        shrinkage = -(param(ii)^2-2*a*lambda*abs(param(ii))+lambda^2)/(2*(a-1));
    elseif (abs(param(ii))>a*lambda)
        shrinkage = (a+1)*lambda^2/2;
    end
    penalty = penalty + shrinkage;
end