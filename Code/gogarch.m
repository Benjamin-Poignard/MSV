function [parameters,ll,Ht] = gogarch(data,p,q,gjrType,type,startingVals,options)
% OGARCH(p,q) and GOGARCH(p,q) multivarate volatility model estimation
%
% USAGE:
%  [PARAMETERS] = rarch(DATA,P,Q)
%  [PARAMETERS,LL,HT,VCV,SCORES] = gogarch(DATA,P,Q,GJRTYPE,TYPE,STARTINGVALS,OPTIONS)
%
% INPUTS:
%   DATA         - A T by K matrix of zero mean residuals -OR-
%                    K by K by T array of covariance estimators (e.g. realized covariance)
%   P            - Positive, scalar integer representing the number of symmetric innovations. Can
%                    also be a K by 1 vector with lag lengths for each series.
%   Q            - Non-negative, scalar integer representing the number of conditional covariance
%                    lags Can also be a K by 1 vector with lag lengths for each series.
 %  GJRTYPE      - [OPTIONAL] Either 1 (TARCH/AVGARCH) or 2 (GJR-GARCH/GARCH/ARCH).  Can also be a K 
%                    by 1 vector containing the model type for each for each series. Default is 2.
%   TYPE         - [OPTIONAL] String, one of 'GOGARCH' (Default) or 'OGARCH'
%   STARTINGVALS - [OPTIONAL] Vector of starting values to use.  See parameters and COMMENTS.
%   OPTIONS      - [OPTIONAL] Options to use in the model optimization (fmincon)
%
% OUTPUTS:
%   PARAMETERS   - Estimated parameters in the order:
%                    OGARCH:
%                    [vol(1) ... vol(K)]
%                    GOGARCH:
%                    [phi(1) ... phi(K(K-1)/2) vol(1) ... vol(K)]
%                    where vol(i) = [alpha(i,1) ... alpha(i,P(i)) beta(i,1) ... beta(i,Q(i))]
%   LL           - The log likelihood at the optimum
%   HT           - A [K K T] dimension matrix of conditional covariances
%   VCV          - A numParams^2 square matrix of robust parameter covariances (A^(-1)*B*A^(-1)/T)
%   SCORES       - A T by numParams matrix of individual scores
%
% COMMENTS:
%   The orthonormal matrix is constructed from K(K-1)/2 angles using PHI2U
%
%
% EXAMPLES:
%   % OGARCH(1,1)
%   parameters = gogarch(data,1,1,[],'OGARCH')
%   % GOGARCH(1,1)
%   parameters = gogarch(data,1,1)
%
% See also BEKK, RARCH, DCC, TARCH, PHI2U

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 1    Date: 4/15/2012

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input Argument Checking
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch nargin
    case 3
        gjrType = [];
        type = [];
        startingVals = [];
        options = [];
    case 4
        type = [];
        startingVals = [];
        options = [];
    case 5
        startingVals = [];
        options = [];
    case 6
        options = [];
    case 7
        % Nothing
    otherwise
        error('3 to 7 inputs required.')
end

if ndims(data)==3
    [k,~,T] = size(data);
else
    [T,k] = size(data);
    temp = zeros(k,k,T);
    for t=1:T
        temp(:,:,t) = data(t,:)'*data(t,:);
    end
    data = temp;
end

if isscalar(p)
    p = ones(k,1)*p;
end
if isscalar(q)
    q = ones(k,1)*q;
end

if isempty(gjrType)
    gjrType = 2;
end
if isscalar(gjrType)
    gjrType = ones(k,1)*gjrType;
end

if isempty(type)
    type = 'gogarch';
end
type = lower(type);
if ~ismember(type,{'gogarch','ogarch'})
    error('TYPE must be either ''GoGARCH'' or ''OGARCH''');
end
if strcmpi(type,'gogarch')
    isGogarch = true;
else
    isGogarch = false;
end


if ~isempty(startingVals)
    count = sum(p) + sum(q);
    if isGogarch
        count = count + k*(k-1)/2;
    end
    if length(startingVals)~=count
        error('STARTINGVALS does not have the correct number of parameters.')
    end
    if isGogarch
        if any(startingVals(1:k*(k-1)/2))>pi || any(startingVals(1:k*(k-1)/2))<0
            error('STARTGINVALS 1 to K*(K-1)/2 must be between 0 and 3.141592')
        end
    end
end

if isempty(options)
  %  options = optimset('fmincon');
    options.Display = 'iter';
    options.Diagnostics = 'on';
    options.Algorithm = 'sqp';
else
    try
        optimset(options);
    catch ME
        error('OPTIONS does not appear to be a valid options structure.')
    end
end
optimoptions.MaxRLPIter = 5000;
optimoptions.MaxFunEvals = 5000;
optimoptions.Algorithm = 'sqp';
optimoptions.TolCon = 1e-09;
optimoptions.TolRLPFun = 1e-09;
optimoptions.MaxSQPIter = 10000;
optimoptions.Diagnostics = 'on';
optimoptions.Jacobian = 'off';
optimoptions.Display = 'iter';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preliminary Estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
S = mean(data,3);
[P,L] = eig(S);
P = P';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Starting Values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isempty(startingVals)
    %startingOptions  =  optimset('fminunc');
%     optimoptions.TolFun = 1e-005;
%     optimoptions.TolX = 1e-005;
%     optimoptions.Display = 'none';
%     optimoptions.LargeScale ='off';
%     optimoptions.MaxFunEvals = 400*(max(p)+max(q));
    
    Zinv = L^(-0.5)*P';
    stdData = zeros(k,k,T);
    for t=1:T
        stdData(:,:,t) = Zinv*data(:,:,t)*Zinv';
    end
    volParams = cell(k,1);
    V = zeros(T,k);
    for i=1:k
        volData = sqrt(squeeze(stdData(i,i,:)));
        [temp,~,V(:,i)] = tarch(volData,p(i),0,q(i),[],gjrType(i),[],optimoptions);
        volParams{i} = temp(2:(1+p(i)+q(i)));
    end
    if isGogarch
        startingVals = zeros(1,k*(k-1)/2)+.0001;
    end
    for i=1:k
        startingVals = [startingVals volParams{i}']; %#ok<AGROW>
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OGARCH
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w = .06*.94.^(0:sqrt(T));
w = w/sum(w);
if isGogarch
    offset = k*(k-1)/2;
else
    offset = 0;
end
parameters = startingVals;

for i=1:k
    if gjrType==1
        backCast = w*abs(volData(1:length(w)));
    else
        backCast = w*(volData(1:length(w))).^2;
    end
    count = p(i) + q(i);
    volStart = parameters(offset + (1:count));
    volData = sqrt(squeeze(stdData(i,i,:)));
     [est,~,~,~,~,~] = fmincon(@(x)ogarch_likelihood(x,volData,p(i),q(i),gjrType(i),backCast),volStart,[],[],[],[],[],[],@(x)constr_ogarch(x),optimoptions);
     parameters(offset + (1:count)) = est;
     offset = offset + count;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GOGARCH
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isGogarch
    v = length(startingVals);
    LB = zeros(1,v);
    UB = LB + 1;
    LB(1:(k*(k-1)/2)) = 1e-6;
    UB(1:(k*(k-1)/2)) = .99998*pi;
    A = zeros(k,v);
    b = .99998*ones(1,k);
    offset = k*(k-1)/2;
    for i=1:k
        count = p(i)+q(i);
        Ai = ones(1,count);
        A(i,offset + (1:count)) = Ai;
        offset = offset + count;
    end
    %parameters = fmincon(@gogarch_likelihood,startingVals,A,b,[],[],LB,UB,[],options,data,p,q,gjrType,P,L,false,false);
    [parameters,~,~,~,~,~] = fmincon(@(x)gogarch_likelihood(x,data,p,q,gjrType,P,L,false,false),startingVals,[],[],[],[],[],[],[],optimoptions);

    [ll,~,Ht] = gogarch_likelihood(parameters,data,p,q,gjrType,P,L,false,false);
else
    [ll,~,Ht] = gogarch_likelihood(parameters,data,p,q,gjrType,P,L,true,false);
end
ll = -ll;
