function somPlotAdjacency(A,W,lineSpec)
% somPlotAdjaceny Draws the som neurons and their adjacencies. 
%
%  somPlotAdjacency(A,W,lineSpec)
%  A       - cell array of size [N,1]. Each cell contains an array of each
%            neuron's neighbors. Such a number of neighbors depends on the 
%            modality.
%  W       - neurons positions: size(W) = [N,p], being p the space
%            dimension.
% lineSpec - line specification. default = 'r.'
%
%__________________________________________________________________________
% Copyright (C) 2014 Signal Processing and Recognition Group
% David C\'ardenas Pe\~na
% $Id: somAdjaceny.m 2014-04-03 00:20:31 $


[N,p] = size(W);
if p > 3
    error('Too many dimensions.')
end

if nargin < 3
  lineSpec = 'r.';
end

figure;
if p==2
  plot(W(:,1),W(:,2),lineSpec)
  hold on
  for i = 1:numel(A)
    for j = 1:numel(A{i})
      plot([W(i,1) W(A{i}(j),1)],[W(i,2) W(A{i}(j),2)],'r')
    end
  end
elseif p==3
  W = W;
end