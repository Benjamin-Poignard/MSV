function output = soft_thresholding(input,lambda)

total = length(input);
output = zeros(total,1);
if length(lambda)==1
   lambda = lambda*ones(total,1);
end
for ii = 1:total
   if input(ii)>lambda(ii)
       output(ii) = input(ii)-lambda(ii);
   elseif abs(input(ii))<=lambda(ii)
       output(ii) = 0;
   elseif input(ii)<-lambda(ii)
       output(ii) = input(ii)+lambda(ii);
   end
end