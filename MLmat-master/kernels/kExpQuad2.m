function k = kExpQuad2(x,y,s)
% kExpQuad Exponentiated quadratic kernel. 
%  k(n,m) = exp( -(x_n-y_m)'*(x_n-y_m)/(2*s^2) ) 
%
%  K = kExpQuad2(X,Y,s) returns a matrix K containing the similarity, using
%      an exponentiated quadratic kernel, between each pair of samples in
%      the data matrix X and the data matrix Y.
%  K = kExpQuad2(X,Y,s)
%  X      - M-by-P data matrix. Rows correspond to observations, columns
%           correspond to variables. 
%  Y      - N-by-P data matrix. Rows correspond to observations, columns
%           correspond to variables. 
%  s      - scale parameter. default: s=median(pdist2(X,Y)).
%  K      - M-by-N kernel matrix
%__________________________________________________________________________
% Copyright (C) 2014 Signal Processing and Recognition Group
% David C\'ardenas Pe\~na
% $Id: kExpQuad.m 2014-03-28 10:37:45 $

if nargin<=1
  error('Not enough input parameters.')
end

x =  pdist2(x,y);

if nargin<=2;
  s = median(x(:));
end

k = exp(-(x.^2)/(2*s^2));
