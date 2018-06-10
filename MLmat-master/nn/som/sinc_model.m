function [P,N]= sinc_model(t,k,f)
%USAGE: 
% [P,V]= sinc_model(t,k,w)
%     t - instant points (numel(t) = N)
%     k - decay rate
%     w - frequency (radians)
%     P - points coordinates (size(P) = [N 2])
%     V - normal for each point (size(V) = [N 2])

if nargin==0
  t = linspace(0,2*pi,100+2);
end
if nargin<1
  k = 1;
end
if nargin<2
  f = 1;
end
t = reshape(t,[numel(t) 1]);
y = exp(-(t/k).^2).*cos(f*t);

P = [t, y];

N = zeros(size(P,1)-2,2);
for i=2:size(P,1)-1
  P1 = P(i-1,:);
  P2 = P(i+1,:);
  v = P1-P2;
  N(i-1,:) = [v(2) -v(1)]/norm(v);
end

P = P(2:end-1,:);

