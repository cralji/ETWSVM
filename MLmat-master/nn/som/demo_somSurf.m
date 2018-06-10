function demo_somSurf()
% demo_somSurf Runs some demos abour the Self-Organizing Map constrained to
%              move along a cloud of points. 
%
%__________________________________________________________________________
% Copyright (C) 2014 Signal Processing and Recognition Group
% David C\'ardenas Pe\~na
% $Id: somAdjaceny.m 2014-04-03 00:29:57 $

clear all;
close all;
clc

writerObj = VideoWriter('/Users/dcardenasp/Documents/som_surf_line.avi');
writerObj.FrameRate = 5;
open(writerObj);


np = 52;
noise = -1 + 2.*rand(1,np);
noise = 2*pi*noise*0.5/np;
[P,N] = sinc_model(linspace(0,2*pi,np)+noise,4,1.5);
np = size(P,1);
%neurons
nw = 10; %number of neurons
A = somAdjacency('line',nw);
W = [linspace(0,2*pi,nw)' zeros(nw,1)];

%learning rate/step length:
eta = 0.1;
for t = 1:10
for i = 1:np
%all distances p_i-w_j
D = pdist2(P,W);
[v,ind_r] = min(D,[],2);
[v,ind_c] = min(D,[],1);

j = ind_r(i); %winner neuron
dir = P(i,:) - W(j,:);
delta_j = eta*dir;
delta_k = zeros(numel(A{j}),2);
for k = 1:numel(A{j})
  dir = computedDirection(W(A{j}(k),:)',...
           P(i,:)', P(ind_c(A{j}(k)),:)',...
           N(ind_c(A{j}(k)),:)', eta);
  delta_k(k,:) = 0.05*D(i,A{j}(k))*dir';
end   
W([j,A{j}],:) = W([j,A{j}],:) + [delta_j;delta_k]; %update
  %plotting stuff:
figure(1)
clf
hold on
  scatter(P(:,1),P(:,2),'b.')
  quiver(P(:,1),P(:,2),N(:,1),N(:,2),0.1)
  scatter(W(:,1),W(:,2),'ro','filled')
  for ii = 1:numel(A)
    for jj = 1:numel(A{ii})
      plot([W(ii,1) W(A{ii}(jj),1)],[W(ii,2) W(A{ii}(jj),2)],'r')
    end
  end
hold off
axis equal
axis([0 2*pi -1.5 1.5])
text(3,-1,['Iteration: ' num2str(t) ' Sample: ' num2str(i)])
drawnow

frame = getframe;
writeVideo(writerObj,frame);
end
end

close(writerObj);
