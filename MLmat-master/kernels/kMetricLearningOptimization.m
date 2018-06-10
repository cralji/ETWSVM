function [eopt,vopt]= kMetricLearningOptimization(k,l,e0)
% Kernel tensor product construction from exponentiated quadratic kernels
% using the equation: T = exp( sum_i( e_i*d_i^2/(2*s_i^2) ) )
% FORMAT [eopt,value] = kTensorProduct(k,l,e0)
% k     - pair-wise kernel similarity. size(d) = [N N P], for N
%         samples and P kernels.
% l     - priori knowledge, size(l) = [N N], for N samples
% e0    - weighting values (e_i>0). size(e) = P, for P kernels
% eopt  - achieved optimum values
% value - objective function value at eopt
%__________________________________________________________________________
% Copyright (C) 2014 Signal Processing and Recognition Group
% David C\'ardenas Pe\~na
% $Id: kMetricLearningOptimization.m 2014-03-27 21:49:00 $

if nargin == 2
  e0 = ones(1,size(k,3));
end
f = @(e)obj_fun(e,k,l);
options = optimoptions('fmincon','GradObj','on','Display','iter');
lb = zeros( size(e0) );
ub = [];%ones(  size(e0) );
[eopt, vopt] = fmincon(f,e0,[],[],[],[],lb,ub,[],options);
vopt = -vopt;

%%%%% objective function %%%%%%%%%%%
function [f,df] = obj_fun(e,K,L)

N         = size(K,1);
P         = size(K,3);
[T,dT_de] = kTensorProduct(K,e);
H         = eye(N)-1/N;
L1        = H*L*H;
T1        = H*T*H;
tr1       = trace( T*L1 );
tr2       = trace( T*T1 );
f         = log( tr1 ) - 0.5*log( tr2 );
f         = -f; % for minimization standards

if nargout > 1
  df_dT  = L1/tr1 - T1/tr2;
  df = zeros(size(e));
  for i = 1:P
    df(i) = trace( df_dT*dT_de(:,:,i) );
  end
  df = -df;%for minimization standards
end