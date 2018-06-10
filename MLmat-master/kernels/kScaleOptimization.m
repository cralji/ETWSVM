function [sopt,vopt]= kScaleOptimization(x,y,s0,obj_func,param)
% Automatic tuning of the scale parameter for the exponentiated quadratic
% kernel: y = exp(d.^2/(2*s^2))
% FORMAT [sigma, value] = kScaleOptimization(X,Y,s0,obj_func,param)
% X        - feature matrix (N x P);
% Y        - feature matrix (M x P);
% s0       - starting point for the search
% obj_func - cost function: 'info' (default). 'var'. More options soon
% param    - if 'info' as obj_func, param is used as the alpha term in the
%            generalized Renyi's entropy function, default=2.
% sigma    - achieved optimum value
% value    - objective function value at sigma
% 
% The tuning method used here is based on the maximization of the
% transformed data variance as a function of the scale parameter, since
% lim_{s->0}{var{y(s)}} = 0 and lim_{s->inf}{var{y(s)}} = 0 and a
% suitable scale value should maximize var{y(s)}
%__________________________________________________________________________
% Copyright (C) 2014 Signal Processing and Recognition Group

% David C\'ardenas Pe\~na
% Andres Marino Alvarez Meza
% $Id: kScaleOptimization.m 2014-02-22 22:40:00 $

if nargin>1
  if numel(y)>0
    x = pdist2(x,y);
  else
    x = pdist2(x,x);
  end
else
  x = pdist2(x,x);      
end

if nargin <= 2
  s0 = median(x(:));
end

func = 'info';
if nargin>3
  func = lower(obj_func);
end

if strcmp(func,'info')
  if nargin>4
    alpha = param;
  else
    alpha = 2;
  end
  f = @(s)info_obj_fun(s,x,alpha);
elseif strcmp(func,'var')
  f = @(s)var_obj_fun(s,x);
else
  error('unknown cost function')
end
[sopt, vopt] = fminsearch(f,s0);


%%%%% information-based objective function %%%%%%%%%%%
function [v] = info_obj_fun(s,x,alpha)

k = exp(-x.^2/(2*s^2));
vi = mean(k,1).^(alpha-1);
v = - var(vi);

%%%%% variance-based objective function %%%%%%%%%%%
function [v] = var_obj_fun(s,x)

k = exp(-x.^2/(2*s^2)); 
k = k(:);
v = -var(k);