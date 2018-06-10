function [X,L] = kPRI(S,starting,beta,gam,opts)
% kPRI computes a compressed version of a dataset matrix using the 
%  Principle of Relevant Information (PRI).
%
%  [X,J] = kPRI(S,Xo,beta,gamma)
%  S     - M-by-P original data matrix. Rows correspond to observations, 
%          columns correspond to variables.
%  Xo    - N-by-P initial compressed data matrix. Rows correspond to 
%          observations, columns correspond to variables.
%  beta  - regularization term between X's entropy and X-S divergence.
%  gamma - parameter for the simulated annealing algorithm used to
%          regularize the scale.
%  X     - N-by-P resulting compressed data matrix. Rows correspond to 
%          observations, columns correspond to variables.
%  J     - Objective function value at each iteration.
%
%  X = kPRI(S,N,beta,gamma)
%  S     - M-by-P original data matrix. Rows correspond to observations, 
%          columns correspond to variables.
%  N     - Number of samples in the compressed data matrix. This number is
%          used to initialize the algorithm.
%  beta  - regularization term between X's entropy and X-S divergence.
%  gamma - parameter for the simulated annealing algorithm used to
%          regularize the scale.
%  X     - N-by-P resulting compressed data matrix. Rows correspond to 
%          observations, columns correspond to variables.
%
%  X = kPRI(S,N,beta,gamma,opts)
%  opts  - options structure:
%          opts.MaxIter = 100;     Maximum number of iterations to perform 
%                                    the algorithm
%          opts.TolX = 1e-4;       Solution gradient stopping criterion 
%          opts.Display = 'iter';  Display results at each iteration 
%                                    {'iter','off'}
%          opts.sigma = 'kso'      Scale parameter for the Parzen estimator
%                                  {scalar,'kso','median'}
% 
% The conventional unsupervised learning algorithms (clustering, principal 
% curves, vector quantization) are solutions to an information optimization 
% problem that balances the minimization of data redundancy with the 
% distortion between the original data and the solution, expressed by
%
% L[p(x|s)] = min H(X) + beta*D(X|S) 
%
% where s in S is the original dataset
%       x in X is a compressed version of the original data achieved 
%              through processing
%       beta   is a variational parameter
%       H(X)   is the entropy 
%       D(X|S) is the KL divergence between the original and the compressed
%              data.
%
% Chapter: Self-Organizing ITL Principles for Unsupervised Learning
% Authors: Sudhir Rao, Deniz Erdogmus, Dongxin Xu, Kenneth Hild II
% Book: Information Theoretic Learning: Renyi's Entropy and Kernel 
%       Perspectives
% Authors: Jose C. Principe
% ISBN: 978-1-4419-1569-6 (Print) 978-1-4419-1570-2 (Online)
%__________________________________________________________________________
% Copyright (C) 2014 Signal Processing and Recognition Group
%
% Andr\'es Eduardo Castro-Ospina
% $Id: kPRI.m 2014-04-11 11:29:31 $

% Number of samples in the original data matrix
m = size(S,1);
% Number of feature in the original data matrix
p = size(S,2);

if numel(starting) == 1
  n  = starting;
  Xo = zeros(n,p);
  mi = min(S); ma = max(S);
  for j = 1:p
    lw = mi(j) + 0.25*(ma(j)-mi(j)) ;
    up = ma(j) - 0.25*(ma(j)-mi(j)) ;
    Xo(:,j) = unifrnd(lw,up,n,1);
  end
else
  Xo = starting;
  n = size(Xo,1);
end
clear starting;

if nargin >= 4
  try
    maxit = opts.MaxIter;
  catch 
    maxit = 100;
  end
  try
    tol   = opts.TolX;
  catch
    tol = 1e-4;
  end
  try 
    show  = opts.Display;
  catch
    show = 'iter';
  end
  try
    sig_alg = opts.sigma;
    if not(ischar(sig_alg))
      sig = sig_alg;
      sig_alg = 'fixed';
    end
  catch
    sig_alg = 'kso';
  end
else
  maxit = 100;
  tol   = 1e-4;  
  show  = 'iter';
  sig_alg = 'kso';
end

% Initial compressed version
X  = Xo;
L  = [];
ii = 0;
NN = tol+1;
if strcmp(show,'iter')
  fprintf('Iter\tObj_fun\t\tGrad_X\n')
end
while NN > tol && ii < maxit
    
  ii = ii + 1;
  X0 = X;
    
  if strcmp(sig_alg,'kso')
    Dsx = pdist2(S,X);
    sig = kScaleOptimization(Dsx);
  elseif strcmp(sig_alg,'median')
    Dsx = pdist2(S,X);
    sig = median(median(Dsx));
%   More scale estimators can be added here.
%   elseif srtcmp(sig_alg,'algnname')
%     sig = f(S,X);
  end  
  sig_n = Multiv_sig(sig,ii,1,gam);
  smin = sig/sqrt(m);
  if sig_n < smin
    sig_n = smin;
  end
  
  % X(t) kernel
  kerX = kExpQuad2(X,X,sig_n);
  %Information potential of X(t)
  IP = sum(kerX(:))/n^2;
  % X,S kernel
  kerXS = kExpQuad2(X,S,sig_n);
  %Cross-information potential
  CIP = sum(kerXS(:))/(n*m);
  
  if isinf(beta)
    c = -m*CIP/(n*IP);
    L(ii) = -log(CIP);
  else
    c = m*CIP/(n*IP)*(1-beta)/2/beta;
    L(ii) = (1-beta)*log(IP) + 2*beta*log(CIP);
  end
  O = ones(m,1);
  O1 = ones(n,1);
        
  DD = repmat(kerXS*O,1,size(X,2));
  
  X = -c*(kerX*X)./DD + (kerXS*S)./DD + c*diag((kerX*O1)./(kerXS*O))*X;

  NN = norm(X0-X,'fro')/norm(X0,'fro');
  
  if strcmp(show,'iter')
    fprintf('%d\t%f\t\t%f\n',ii,L(ii),NN)
  end
    
end

function sig_n = Multiv_sig(sig0,n,k1,Z)
if isinf(Z)
  sig_n = (k1*sig0)/(1+k1*n);
else
  sig_n = (k1*sig0)/(1+Z*k1*n);
end
