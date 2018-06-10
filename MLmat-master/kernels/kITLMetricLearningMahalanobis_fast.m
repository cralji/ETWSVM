function [A,K] = kITLMetricLearningMahalanobis_fast(X,L,...
    alpha,etav,plot_it,A_i,labels)
% Dimensionality reduction using gramm matrices based metric learning
% Function basics
%Sanchez et. al. Information theoretic learning with infitely divisible kernels
%Cardenas & Alvarez  Sigma tune Gaussian kernel with information potential
% USAGE:
% [A,K] = kITLMetricLearningMahalanobis_fast(X,L,...
%    alpha,etav,plot_it,A_i,labels)
% Inputs:
%X \in R^{N x P} : high dimensional data matrix, N: samples; p:features
%L \in R^{N x N} : high-dimensional kernel
%alpha \in R : Renyi entropy order
%etav \in R^2 : iteration step size
%labels \in Z^{N x 1} group membership if provided
%A_i \in R^{P x Q} : Rotation matrix for mahalanobis distance dA(x,x') = d(xA,x'A)

%plot_it : if true every xx iterations the gradient based results are shown
% Output:
% A \in R^{P x Q} learned rotation matrix by minimizing conditional entropy between high dimensional and low dimensional samples
% K \in R^{N x N} computed kernel in low dimensional space 
%K_A : is computed using mahalanobis distance into a gaussian kernel, which kernel band-width
% is fixed by maximizing a information potential variance based function (see kscaleOptimization)
%%
if nargin < 3
    alpha = 2;
    etav = [0.001 0.001];
    labels = 1 : size(X,1);
    plot_it = true;
    Q = 2;
    if size(X,2)<Q
        Q = size(X,2);
    end
    [~,A_i] = ml_pca(X,Q); %pca based
else if nargin < 4
        etav = [0.001 0.001];
        labels = 1 : size(X,1);
        plot_it = true;
        Q = 2;
        if size(X,2)<Q
            Q = size(X,2);
        end
        [~,A_i] = A_pca(X,Q); %pca based
    else if nargin < 5
            labels = 1 : size(X,1);
            plot_it = true;
            Q = 2;
            if size(X,2)<Q
                Q = size(X,2);
            end
            [~,A_i] = A_pca(X,Q); %pca based
            
        else if nargin < 6
                labels = 1 : size(X,1);
                plot_it = true;
                Q = 2;
                if size(X,2)<Q
                    Q = size(X,2);
                end
                [~,A_i] = A_pca(X,Q); %pca based
            else if nargin < 7
                    labels = 1 : size(X,1);
                end
            end
        end
    end
end
%% optimization set up
tol = 1e-4;
maxiter = 100;
eta_start=etav(1);
eta_end=etav(2);
fnew = inf;

%% scaling rotation matrix based on information potential variability maximization
sopt = kScaleOptimization_info(pdist(X*A_i));
s0 = 1/(2*sopt^2);
A =  sqrt(s0)*A_i;

vecAs = A(:);
N = size(X,1);
%H=eye(N)-1/N*ones(N,1)*ones(1,N);
%Normalization to have unit trace
L = L./trace(L);

%ojo
%L = (1/N)*H*L*H;
%L = H*L*H;
%L = (1/N)*L;

if plot_it
    figure(2)
    title('Projection...')
    
    figure(1)
    title('Cost function')
    xlabel('iteration')
    ylabel('Centered alignment')
    hold on
    %showfigs_c(2);
end
%%
for ii=1:maxiter
    fold = fnew;
    
    if ii<maxiter/4
        eta=eta_start;
    else
        eta=eta_end;
    end
    [fnew,gradf,K,Y]=derivativeAs_ITL(vecAs,X,N,L,alpha);
    % do linesearch
    vecAs=vecAs-eta*gradf;
    df = abs(fold-fnew);
    if plot_it
        fprintf('it%d-%d-f=%.2e-diff_f=%.2e-eta=%.2e\n',...
            ii,maxiter,fold,df,eta)
        figure(1)
        scatter(ii,fnew,20,'r','filled')
        drawnow
    end
    if plot_it == true && mod(ii,1) == 0 || ii == 1 && plot_it == true
        figure(2)
        clf
        subplot(2,2,4)
        %scatter(Y(:,1),Y(:,2),20,labels,'filled');
        axis off
        title('Y')
        subplot(2,2,3)
        imagesc(K), axis square, colorbar
        axis off
        title('K')
      
        subplot(2,2,1)
        imagesc(L), axis square, colorbar
        axis off
        title('L')
        
        subplot(2,2,2)
        imagesc(real(reshape(vecAs,size(A)))), colorbar
        title('A')
        axis off
        drawnow
    end
    
    if df < tol
        fprintf('Metric Learning done...diffi %.2e= \n',df)
        break;
    end
    
end

A=real(reshape(vecAs,size(X,2),[]));

if plot_it == true
    figure(2)
    hold off
end

end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f, gradf,k,y] = derivativeAs_ITL(vecas,x,n,l,alpha)
a=real(reshape(vecas,size(x,2),[]));
y=x*a;
d=pdist2(y,y);
%sigma tune
%[sopt]= kScaleOptimization_info(d);
%k=(1/n)*exp(-d.^2./(2*sopt^2));
k=exp(-d.^2);
k = k./trace(k);
%k = h*k*h;
%k=exp(-d.^2/(2*sopt^2));

if any(isnan(k(:)))% ||any(isnan(k_ly(:)))
    fprintf('whoa')
    f=nan;
else
    gsa = gsalpha(n*k.*l,alpha);
    gsak = gsalpha(k,alpha);
    P = (n*l.*gsa - gsak).*k;
    gradf = real(x'*(P-diag(P*ones(n,1)))*x*a);
    gradf = gradf(:);
    f = salpha(k.*l,alpha)-salpha(k,alpha);
end
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function gs_a = gsalpha(am,alpha)
[U,D] = eigs(am,rank(am));
gs_a = (alpha*U*D^(alpha-1)*U')/((1-alpha)*trace(am^alpha));
%gs_a = alpha/(1-alpha)*(am^(alpha-1)/trace(am^alpha));
end
%%
function s_a = salpha(am,alpha)
s_a = (1/(1-alpha))*log(trace(am^alpha));
end