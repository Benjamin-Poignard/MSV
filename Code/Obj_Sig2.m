function obj = Obj_Sig(theta,Sig_x,Sig_u,mPhi,mXi,iT)
iN = length(Sig_x);
mA = tril(dvech(theta,iN));Sig_alpha = mA*mA';
Sig_zeta = -0.5*(inv(mPhi)*mXi*Sig_u+Sig_u*mXi'*inv(mPhi'));
S_a = Sig_x -Sig_zeta;
%obj = 0.5*iN*log(det(Sig)) + 0.5*trace(inv(Sig)*Sig_x);
%obj = 0.5*iN*2*sum(log(diag(mA))); + 0.5*trace(inv(Sig)*Sig_x);
mD = Sig_alpha-S_a;
obj = trace(mD*mD');
end
 