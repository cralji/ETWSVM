function u = mklLinearCombination(k,l)

N=size(k,1);
P=size(k,3);
H=eye(N)-1/N;
L1=H*l*H;
K1 = zeros(size(k));
a  = zeros(1,P);
for p=1:P
  K1(:,:,p)=H*k(:,:,p)*H;
  a(p)=trace(K1(:,:,p)*L1');
end
M=mklComputeM(K1);
u=(M\a')./norm(M\a');