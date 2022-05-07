function B = positivity(A,N,transform)

dim = N*(N+1)/2;

switch transform
    case 'check_pos'
        block = [];
        parfor ii = 1:dim
            K_temp = dvech(A(ii,:),N);
            K_temp = tril(K_temp,-1)/2+tril(K_temp,-1)'/2+diag(diag(K_temp));
            block = [block;K_temp];
        end
        B = zeros(N^2,N^2);
        for ii = 1:N
            for kk = ii:N
                B(1+(ii-1)*N:ii*N,1+(kk-1)*N:kk*N) = block(1:N,:);
                block(1:N,:) = [];
            end
        end
        for ii = 1:N^2
            for kk = 1:N^2
                B(kk,ii)=B(ii,kk);
            end
        end
    case 'vech'
        block = [];
        for ii = 1:N
            for kk = ii:N
                block = [block;A(1+(ii-1)*N:ii*N,1+(kk-1)*N:kk*N)];
            end
        end
        block_temp = zeros(N,N,dim); B = zeros(dim,dim);
        parfor ii = 1:dim
            K_temp = block(1+(ii-1)*N:ii*N,:);
            K_temp = 2*tril(K_temp,-1)+2*tril(K_temp,-1)'+diag(diag(block(1+(ii-1)*N:ii*N,:)));
            block_temp(:,:,ii) = K_temp;
            B(ii,:) = vech(block_temp(:,:,ii))';
        end
end