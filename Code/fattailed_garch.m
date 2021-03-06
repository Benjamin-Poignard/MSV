function [parameters, likelihood, stderrors, robustSE, ht, scores] = fattailed_garch(data , p , q , errors, startingvals, options)

% Inputs:
%     -data: A single column of zero mean random data, normal or not for quasi likelihood
%     -P: Non-negative, scalar integer representing a model order of the ARCH 
%       process
%     -Q: Positive, scalar integer representing a model order of the GARCH 
%       process: Q is the number of lags of the lagged conditional variances included
%       Can be empty([]) for ARCH process
%     -error:  The type of error being assumed, valid types are:
%            'NORMAL' - Gaussian Innovations
%            'STUDENTST' - T-distributed errors
%            'GED' - General Error Distribution
%  
%     -startingvals: A (1+p+q) (plus 1 if STUDENTT OR GED is selected for the nu parameter) vector of starting vals.
%       If you do not provide, a naieve guess of 1/(2*max(p,q)+1) is used for the arch and garch parameters,
%       and omega is set to make the real unconditional variance equal
%       to the garch expectation of the expectation.
%     -options: for fmincom: NEED THE PACKAGE
% Outputs:
%     -parameters : a [1+p+q X 1] column of parameters with omega, alpha1, alpha2, ..., alpha(p)
%                  beta1, beta2, ... beta(q)
%     -likelihood = the loglikelihood evaluated at he parameters
%     -robustSE = Bollersev Wooldridge
%     -stderrors = the inverse analytical hessian, not for quasi maximum liklihood
%     -ht = the estimated time varying VARIANCES
%     -scores = The numberical scores(# fo params by t) for M testing   

t=size(data,1);
if nargin<6
    options=[];
end

if strcmp(errors,'NORMAL') | strcmp(errors,'STUDENTST') | strcmp(errors,'GED')
   if strcmp(errors,'NORMAL') 
      errortype = 1;
   elseif strcmp(errors,'STUDENTST') 
      errortype = 2;
   else
      errortype = 3;
   end
else
   error('error must be one of the three strings NORMAL, STUDENTST, or GED');
end


if size(data,2) > 1
   error('Data series must be a column vector.')
elseif isempty(data)
   error('Data Series is Empty.')
end


if (length(q) > 1) | any(q < 0)
   error('Q must ba a single positive scalar or 0 for ARCH.')
end

if (length(p) > 1) | any(p <  0)
   error('P must be a single positive number.')
elseif isempty(p)
   error('P is empty.')
end

if isempty(q) | q==0;
   q=0;
   m=p;
else
   m  =  max(p,q);   
end


if nargin<=4 | isempty(startingvals)
   guess  = 1/(2*m+1);
   alpha  =  .15*ones(p,1)/p;
   beta   =  .75*ones(q,1)/q;
   omega  = (1-(sum(alpha)+sum(beta)))*cov(data);  
   if strcmp(errors,'STUDENTST')
      nu  = 30;
   elseif strcmp(errors,'GED')
      nu = 1.7;
   else
      nu=[];
   end
else
   omega=startingvals(1);
   alpha=startingvals(2:p+1);
   beta=startingvals(p+2:p+q+1);
   if strcmp(errors,'STUDENTST')
      nu  = startingvals(p+q+2);
   elseif strcmp(errors,'GED')
      nu = startingvals(p+q+2);
   else
      nu=[];
   end
end


LB         =  [];     
UB         =  [];     
sumA =  [-eye(1+p+q); ...
      0  ones(1,p)  ones(1,q)];
sumB =  [zeros(1+p+q,1);...
      1];                          


if (nargin <= 5) | isempty(options)
   options  =  optimset('fmincon');
   options  =  optimset(options , 'TolFun'      , 1e-006);
   options  =  optimset(options , 'Display'     , 'iter');
   options  =  optimset(options , 'Diagnostics' , 'on');
   options  =  optimset(options , 'LargeScale'  , 'off');
   options  =  optimset(options , 'MaxFunEvals' , 400*(2+p+q));
end

sumB = sumB - [zeros(1+p+q,1); 1]*2*optimget(options, 'TolCon', 1e-6);

if strcmp(errors,'STUDENTST')
   LB = zeros(1,p+q+2);
   LB(length(LB))=2.1;
   [n,M]=size(sumA);
   sumA = [sumA';zeros(1,n)]';
elseif strcmp(errors,'GED')
   LB = zeros(1,p+q+2);
   LB(length(LB))=1.1;
   [n,M]=size(sumA);
   sumA = [sumA';zeros(1,n)]';
else
   LB = [];
end



if errortype == 1
   startingvals = [omega ; alpha ; beta];
else
   startingvals = [omega ; alpha ; beta; nu];
end

% Estimate the parameters.
stdEstimate =  std(data,1);  
data        =  [stdEstimate(ones(m,1)) ; data];  
T=size(data,1);

[parameters, LLF, EXITFLAG, OUTPUT, LAMBDA, GRAD] =  fmincon('fattailed_garchlikelihood', startingvals ,sumA  , sumB ,[] , [] , LB , UB,[],options, data, p , q, errortype, stdEstimate, T);


if EXITFLAG<=0
   EXITFLAG
   fprintf(1,'Not Sucessful! \n')
end

parameters(find(parameters    <  0)) = 0;          
parameters(find(parameters(1) <= 0)) = realmin;    
hess = hessian_2sided('fattailed_garchlikelihood',parameters,data,p,q,errortype, stdEstimate, T);
[likelihood, ht]=fattailed_garchlikelihood(parameters,data,p,q,errortype, stdEstimate, T);
likelihood=-likelihood;
stderrors=hess^(-1);

if nargout > 4
   h=max(abs(parameters/2),1e-2)*eps^(1/3);
   hplus=parameters+h;
   hminus=parameters-h;
   likelihoodsplus=zeros(t,length(parameters));
   likelihoodsminus=zeros(t,length(parameters));
   for i=1:length(parameters)
      hparameters=parameters;
      hparameters(i)=hplus(i);
      [HOLDER, HOLDER1, indivlike] = fattailed_garchlikelihood(hparameters,data,p,q,errortype, stdEstimate, T);
      likelihoodsplus(:,i)=indivlike;
   end
   for i=1:length(parameters)
      hparameters=parameters;
      hparameters(i)=hminus(i);
      [HOLDER, HOLDER1, indivlike] = fattailed_garchlikelihood(hparameters,data,p,q,errortype, stdEstimate, T);
      likelihoodsminus(:,i)=indivlike;
   end
   scores=(likelihoodsplus-likelihoodsminus)./(2*repmat(h',t,1));
   scores=scores-repmat(mean(scores),t,1);
   B=scores'*scores;
   robustSE=stderrors*B*stderrors;
end
