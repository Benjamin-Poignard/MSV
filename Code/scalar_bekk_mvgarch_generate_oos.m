function [LLF,likelihoods,Ht]=scalar_bekk_mvgarch_generate_oos(parameters,errors,errors_in,p,q,k,k2,t)

C=parameters(1:(k2));
A=parameters(k2+1:k2+p);
B=parameters(k2+p+1:k2+p+q);

C=ivech(C);
C=tril(C);
const=C*C';

uncond=cov(errors_in);
m=1;
eeprime=zeros(k,k,t+m);
Ht=zeros(k,k,t+m);
for i=1:m
    eeprime(:,:,i)=uncond;
    Ht(:,:,i)=uncond;
end
LLF=0;
errors=[repmat(sqrt(diag(uncond))',m,1);errors];
likelihoods=zeros(t+m,1);
for i=m+1:t+m;
    Ht(:,:,i)=const;
    for j=1:p
        Ht(:,:,i)=Ht(:,:,i)+A(j)*(errors(i-j,:))'*(errors(i-j,:))*A(j);
    end
    for j=1:q
        Ht(:,:,i)=Ht(:,:,i)+B(j)*Ht(:,:,i-j)*B(j);
    end
    % trick to ensure positive-definiteness, when the dimension is high
    if (eig(Ht(:,:,i))<0.01)
        Ht(:,:,i) = Ht(:,:,i)+0.01*eye(k);
    end
    
    likelihoods(i)=k*log(2*pi)+(log(det(Ht(:,:,i)))+errors(i,:)*Ht(:,:,i)^(-1)*errors(i,:)');
    LLF=LLF+likelihoods(i);
end
LLF=0.5*(LLF);
likelihoods=0.5*likelihoods(m+1:t+m);
Ht=Ht(:,:,m+1:t+m);
if isnan(LLF)
    LLF=1e6;
end


