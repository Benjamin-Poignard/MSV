function [c,ceq] = nlcon_Sig(theta,Sig_x,Sig_u,mPhi,mXi,iT)
%  OUTPUTS:

%SolveSigma(x(:,p+1:end)',Sig_u,Phi_sec,Xi_sec)

%    C    - Vector of nonlinear inequality constraints.  
%    CEQ  - Empty matrix
iN = length(Sig_x);
mA = tril(dvech(theta,iN));Sig_alpha = mA*mA';
[~,mD] = eig(Sig_x-Sig_alpha);
iS= min(diag(mD)); iM = max(diag(mD));
gam = iN/iT; iA = (1-1/sqrt(gam))^2; iB = (1+1/sqrt(gam))^2;
%c = -iS+0.01;%c
c = - iS/iM + iA/iB;
ceq=[];
end