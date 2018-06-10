function A = somAdjacency(mode,N)
% somAdjaceny Self-Organizing Map adjancency matrix for network's neurons. 
%
%  A = somAdajacency(mode,N) returns a cell A containing the index of the
%      neighbors of each neuron.
%  mode   - Adjacency modality: 'line', 
%  N      - Number of neurons.
%  A      - cell array of size [N,1]. Each cell contains an array of each
%           neuron's neighbors.Such a number of neighbors depends on the 
%           modality.
%
%__________________________________________________________________________
% Copyright (C) 2014 Signal Processing and Recognition Group
% David C\'ardenas Pe\~na
% $Id: somAdjaceny.m 2014-04-03 00:05:39 $

switch mode
case 'line'
  tmp = [1:N-2;3:N]';
  tmp = mat2cell(tmp,ones(N-2,1),2);
  A{1} = 2;
  A(2:N-1) = tmp;
  A{N} = N-1;
otherwise
  error('Unknown modality.')
end