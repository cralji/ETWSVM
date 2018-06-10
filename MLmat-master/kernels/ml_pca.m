%PCA
%Xp: (nxp)-sized input matrix
%d: If d>1: Number of output components. If 0<d<1 percentage of kept variance

function [Y, W, Val, Vec]=ml_pca(X,d)


[n, p]=size(X);
Xpp=X;

if n>p
    P=Xpp'*Xpp; %outter product pxp
else
    P=Xpp*Xpp'; %inner product nxn
end

[Vec, Val]=eig(P);
Val = abs(diag(Val));
Val = Val./sum(Val);
%sort from largest to smallest
[Val, ival]=sort(Val,'descend');
Vec = Vec(:,ival);

if d == 0 %L-curve
    x = linspace(0,1,numel(Val))';
    y = Val/max(Val);
    [~,ind]=min(x.^2 + y.^2);
    ivald = 1:ind;
%     ivald = Val >= mean(Val);
    W = Vec(:,ivald);

elseif d > 0 && d < 1
   
    va = 0;
    W = [];
    i = 1;
    while va < d
        W = [W,Vec(:,i)];
        va = va+Val(i);
        i=i+1;
    end

elseif d >=  1
    W = Vec(:,1:min(d,size(Vec,2)));
end

if n<p
    W = Xpp'*W*diag(Val(1:size(W,2)).^-.5);
    %W = W*diag(Val(1:size(W,2))).^-.5;
end
Y = Xpp*W;
