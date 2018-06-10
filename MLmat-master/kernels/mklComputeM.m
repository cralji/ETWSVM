function m = mklComputeM(K)
m=zeros(size(K,3),size(K,3));
for p=1:size(K,3)
  for l=1:size(K,3)
    m(p,l)=trace(K(:,:,p)*K(:,:,l));
  end
end