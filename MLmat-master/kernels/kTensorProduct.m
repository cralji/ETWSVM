function [T,dT] = kTensorProduct(k,e)
% Kernel tensor product construction from exponentiated quadratic kernels
% using the equation: T = exp( sum_i( e_i*d_i^2/(2*s_i^2) ) )
% FORMAT [T,dT_de] = kTensorProduct(k,e)
% d     - pair-wise kernel similarity. size(d) = [N N P], for N
%         samples and P kernels.
% e     - weighting values (e_i>0). size(e) = P, for P kernels
% T     - resulting tensor kernel. size(T) = [N N]
% dT_de - derivative wrt the weighting values, for optimization purposes.
%         size(dTde) = [N N P], for P kernels
%__________________________________________________________________________
% Copyright (C) 2014 Signal Processing and Recognition Group

% David C\'ardenas Pe\~na
% $Id: kTensorProduct.m 2014-03-27 20:06:00 $

N = size(k,1);
P = size(k,3);

T = k(:,:,1).^e(1);
for i = 2:P
  T = T.*k(:,:,i).^e(i);
end

if nargout == 2
  dT = zeros([N N P]);
  for i = 1:P
    dT(:,:,i) = log(k(:,:,i)).*T;
  end
end