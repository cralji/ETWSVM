function k = kExpQuad(x,s,mode)
% kExpQuad Exponentiated quadratic kernel. 
%  k(n,m) = exp( -(x_n-x_m)'*(x_n-x_m)/(2*s^2) ) 
%  k(i)   = exp( -d_i.^2/(2*s^2) ) 
%
%  K = kExpQuad(X,s) returns a matrix K containing the similarity, using
%      an exponentiated quadratic kernel, between each pair of 
%      observations.
%  K = kExpQuad(X,s,'features') returns a matrix K containing the 
%      similarity, using an exponentiated quadratic kernel, assuming X as a
%      feature matrix.
%  X      - M-by-N data matrix. Rows correspond to observations, columns
%           correspond to variables. 
%  s      - scale parameter. default: s=median(pdist(x)).
%  K      - M-by-M kernel matrix
%
%  K = kExpQuad(d,s,'distances') returns a matrix K containing the 
%      similarity, using an exponentiated quadratic kernel, between each 
%      pair of observations.
%  d      - M-by-M pairwise distance matrix or M*(M-1)/2 vector of
%           distances, for M observations.
%  s      - scale parameter. default: s=median(d(:)).
%  K      - M-by-M kernel matrix
%
%__________________________________________________________________________
% Copyright (C) 2014 Signal Processing and Recognition Group
% David C\'ardenas Pe\~na
% $Id: kExpQuad.m 2014-03-28 10:37:45 $

if nargin<=2;
  mode = 'features';
end

if strcmp(mode,'features')
  x =  pdist(x);
elseif strcmp(mode,'distances')
  if size(x,1) == size(x,2)
    x = squareform(x);
  end
else
  error('Unknown mode.')  
end

if nargin==1
  s = median(x(:));
end

k = exp(-squareform(x.^2)/(2*s^2));
