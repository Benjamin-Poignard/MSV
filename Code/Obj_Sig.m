function obj = Obj_Sig(theta,Sig_x,Sig_u,mPhi,mXi)
iN = length(Sig_x);iNs = 0.5*iN*(iN+1);
mA = tril(dvech(theta(1:iNs),iN));Sig_alpha = mA*mA';
mB = tril(dvech(theta(iNs+1:end),iN));Sig_zeta = mB*mB';

mS1 = (Sig_alpha + mPhi*Sig_alpha*mPhi')+(Sig_zeta+mPhi*Sig_zeta*mPhi');
mS2 = Sig_u + mXi*Sig_u*mXi';
mD1 = mS1-mS2;
mD2 = (-mPhi*Sig_zeta)-mXi*Sig_u;
mD3 = Sig_x - (Sig_alpha+Sig_zeta);
vD = [vec(mD1); vec(mD2); vec(mD3)];
obj = vD'*vD;
end
 