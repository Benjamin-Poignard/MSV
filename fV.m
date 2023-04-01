function mV = fV(mS,mP,iT)
  mQ2 = mS; iN = length(mS);
  mQ1 = {mS}; 
  for i=2:iT
      mQ2 = mP*mQ2;
      mQ1(:,i) = {mQ2};
  end
  mQ3 = cell2mat(mQ1(toeplitz(1:iT)));
  mQ4 = tril(mQ3);
  mV = dvech(vech(mQ4),iT*iN);
end


