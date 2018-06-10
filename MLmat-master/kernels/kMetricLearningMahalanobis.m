function [A,s,K,F,exitflag] = kMetricLearningMahalanobis(X,L,labels,opts)
% metric learning - mahalanobis distance
% Function basics
%Brockmeier et. al. Neural Decoding with kernel-based metric learning
%Cardenas & Alvarez  Sigma tune Gaussian kernel with information potential
%USAGE:
% [A,s,K,J,exitflag] = kMetricLearningMahalanobis(X,L,labels,options)
%INPUTS:
% X \in R^{N x P}      : Data matrix, N: samples; P:features
% L \in R^{N x N}      : Kernel for aligment
% labels \in Z^{N x 1} : Group membership
% options              : Options structure (optional). See ckaoptimset:
% OUTPUTS:
% A \in R^{P x Q}      : Learned rotation matrix by maximizing centered 
%                        kernel alignment -> log(rho(K_A,L))
% s \in Real^+         : Kernel bandwidth
% K \in R^{N x N}      : Output kernel
% J \in Z^{T x 2}      : Cost function values at each iteration for train
%                        and validation sets.
% exitflag             : Reason algorithm stopped:
%                        0: Minimum gradient magnitude achieved
%                        1: Validation checks exceeded.
%                        2: Minimum goal achieved. 
%                        3: Maximum iterations exceeded.                      
%                       -1: Negative eigenvalues obtained.
% K_A : is computed using mahalanobis distance ||A*x_i - A*x_j|| into a 
% gaussian kernel, which bandwidth is fixed by maximizing an information 
% potential variance based function (see kscaleOptimization).
%Version 2.0 Changes:
% -Stopping criteria included using training and validation groups.
% -Option structure for generality and modularity.

%% Parsing arguments
N = size(X,1);

if nargin>3
  options = ckaoptimset(opts);
else
  options = ckaoptimset();
end

Q = options.Q;
if strcmp(options.init,'pca')
  [~,A_i] = ml_pca(X,Q); %pca based
else
  A_i = options.init;
end

if strcmp(options.training,'default')
  trInd = true(N,1);
else
  trInd = options.training;  
end
valInd = ~trInd;

if strcmp(options.maxiter,'default')
  maxiter = 10*numel(A_i);
else
  maxiter = options.maxiter;
end

etav     = options.lr;
% print_it = options.showCommandLine;
% plot_it  = options.showWindow;
print_it=false;
plot_it=false;

min_grad = options.min_grad;
goal     = options.goal;
max_fail = options.max_fail;
    
%% Initialization

Xval       = X(valInd,:);
X          = X(trInd,:);
Lval       = L(valInd,valInd);
L          = L(trInd,trInd);
labels_val = labels(valInd);
labels     = labels(trInd);
Nval       = sum(valInd);
N          = sum(trInd);

%For training data:
sopt = kScaleOptimization(X*A_i);
A = A_i/(sqrt(2)*sopt);
s0 = 1/(2*sopt^2);
u_i = log(s0)/log(10);

vecAs = [A(:);u_i];
H=eye(N)-1/N*ones(N,1)*ones(1,N);
Hval=eye(Nval)-1/Nval*ones(Nval,1)*ones(1,Nval);

eta_start=etav(1);
eta_end=etav(2);
F = zeros(maxiter,2);
fnew = inf;
fbest = inf;
checks = 0;
exitflag = 3;

if plot_it
  fig2 = figure(2);
  title('Projection...')    
  fig1 = figure(1);
  title(['Checks ' num2str(checks)])
  xlabel('iteration')
  ylabel('Centered Alignment')
  hold on
%     showfigs_c(2);
end

%% Optimization
for ii = 1 : maxiter
    fold = fnew;
    
    [fnew,gradf,K,Y] = A_derivativeAs(vecAs,X,N,L,H);
    F(ii,1) = fnew;
    if Nval>0
      [F(ii,2),~,K,Y] = A_derivativeAs(vecAs,Xval,Nval,Lval,Hval);
      labels = labels_val;
      if F(ii,2)>fbest
        checks = checks + 1;
      else
        checks = 0;
        fbest = F(ii,2);
      end
    end
    if fnew > fold %last A
        %fnew = fold;
        eta_start = eta_start-eta_start*.1;
        eta_end = eta_end-eta_end*.25;
    end
    if ii < maxiter/2
        eta = eta_start;
    else
        eta = eta_end;
    end

    %Eigenvalue check:
    if abs(norm(gradf)-norm(real(gradf))) > 0
      exitflag = -1;
      if print_it
        fprintf('Negative eigenvalues found.\n')
      end
      break;
    end
    
    % Do linesearch:
    dg = norm(gradf);    
    vecAs = vecAs - eta*gradf;
%     df = abs(fold-fnew);
      
    if print_it && (mod(ii,10) == 0 || ii == 1)
       fprintf('%d-%d -- eta = %.2e -- f = %.2e -- |df_dx| = %.2e\n',...
                ii,maxiter,eta,fnew,dg)
    end
    if plot_it
        set(0, 'currentfigure', fig1)
        scatter(ii,F(ii,1),20,'b','filled')
        if Nval>0
          scatter(ii,F(ii,2),20,'g','filled')
          legend('Training','Validation')
        end
        title(['Checks ' num2str(checks)])
        drawnow
    end
    if plot_it && (mod(ii,10) == 0 || ii == 1)
        set(0, 'currentfigure', fig2)
        clf
        subplot(2,2,4)
        if size(Y,2)==2
          scatter(Y(:,1),Y(:,2),20,labels,'filled')
        elseif size(Y,2)>2
          scatter3(Y(:,1),Y(:,2),Y(:,3),20,labels,'filled')
        end
        axis off
        title('Y')
        
        subplot(2,2,3)
        imagesc(K); axis square; colorbar
        axis off
        title('K')
        
        subplot(2,2,1)
        imagesc(L); axis square; colorbar
        axis off
        title('L')
        
        subplot(2,2,2)
        imagesc(real(reshape(vecAs(1:end-1),size(A)))), colorbar
        title(['A' ' - \sigma = ' num2str(sqrt(1/(2*10^vecAs(end))),'%.2e')])
        axis off
        drawnow
    end
    
    %Checking stopping criteria
    if dg < min_grad
      exitflag = 0;
      if print_it
        fprintf('Metric Learning done...Gradient norm = %.2e\n',dg)
      end
      break;
    end
    
    if checks>=max_fail
      exitflag = 1;
      if print_it
        fprintf('Metric Learning done... Fails = %d\n',checks)
      end      
      break;
    end
        
    if fnew < goal
      exitflag = 2;
      if print_it
        fprintf('Metric Learning done...Goal = %.2e\n',fnew)
      end
      break;
    end    
end

A=real(reshape(vecAs(1:end-1),size(X,2),[]));
sp = kScaleOptimization(X*A);
A = A./(sqrt(2)*sp);
s = vecAs(end);
s = 10^s;

F = F(1:ii,:);
if plot_it
  set(0, 'currentfigure', fig2)
  hold off
end

end
%% cka derivative
function [f, gradf,k,y,rho]= A_derivativeAs(vecas,x,n,l,h)
a=real(reshape(vecas(1:end-1),size(x,2),[]));
u = vecas(end);
s = 10^u;
sp = kScaleOptimization(x*a);
a = a/(sqrt(2)*sp);
y=x*a;
d=pdist2(y,y);
k=exp(-d.^2/2);

if any(isnan(k(:)))% ||any(isnan(k_ly(:)))
    fprintf('whoa')
    f=nan;
    gradf = zeros(size(vecas));
    rho=nan;
    %    gradf=0*logeta;
else
    trkl=trace(k*h*l*h);
    trkk=trace(k*h*k*h);
    
    grad_lk=(h*l*h)/trkl;
    grad_k=2*(h*k*h)/trkk;
    grad = grad_lk-.5*grad_k;
    p = grad.*k;
    
    p=(p+p')/2;
    grada = x'*(p-diag(p*ones(n,1)))*(x*a);
    grada = -4*real(grada(:));
    grads = trace((-k.*d.^2)*real(grad));
    grads = s*log(10)*grads; %function of u; s = 10^u
    
    gradf= real([grada;grads]);
    
    f=-real(log(trkl)-log(trkk)/2);
    rho = trkl/sqrt(trkk);
end

end