clc
close all
clear

%% AddPaths
addpath(genpath('MLmat-master'));
addpath(genpath('ETWSVM_Tools'));

%% GenerateData
%% Example 1
% n = 150;
% X = [randn(n,2); randn(n,2)];
% X(1:n,1) = X(1:n,1) + 8;
% % X = zscore(X);
% y = [-ones(n,1);ones(n,1)]; % clase 1 - clase 2
% ParFor
% nc=2;
% matlabpool close force
% matlabpool(nc); 
% 
% BaseDatos='dona';
% load(['data/',BaseDatos])
% % Data=load(['data/',BaseDatos]);
% X=Data(:,1:end-1);
% y=Data(:,end);
% X3 = 2*X(y==-1,:) ;
% %X4 = 4*X(y==-1,:) ;
% %X = [X;X3;X4];
% X = [X;X3];
%y = [y;ones(sum(y==-1),1); -ones(sum(y==-1),1)];
% y = [y;ones(sum(y==-1),1)]*-1;


% % % %  Example 2
Data=load('data/SinteticoDona1.txt');
X=Data(:,1:end-1);
y=Data(:,end);

X = zscore(X);
figure(1)
scatter(X(:,1),X(:,2),30,y,'filled')

%% Gaussian kernel parameter grid


so=median(pdist(X));
Vs=linspace(0.01*so,so,10)';

clear exp1 exp2;
%% contour plot
xs = std(X(:,1));
ys = std(X(:,2));
ux = mean(X(:,1));
uy = mean(X(:,2));
lix = ux - 3.5*xs;
lsx = ux + 3.5*xs;
liy = uy - 3*ys;
lsy = uy + 3*ys;

np = 500;
Xm = meshgrid(lix: (lsx - lix)/np :lsx);
Ym = meshgrid(liy: (lsy - liy)/np :lsy);
Ym=Ym';
Xmv=Xm(:);
Ymv=Ym(:);
TestX=[Xmv,Ymv];
Fm = zeros(np+1,np+1);
label_estc = zeros(np+1,np+1);

%% Kernel Function
kern.kernfunction='rbf';
tipo = 'nolineal';
%% Metric learning -- CKA    
opts = ckaoptimset();
opts.showCommandLine=false;
opts.showWindow=false;
opts.maxiter = 100;
opts.Q =0.99;  
[~,~,label] = SeparacionClases(X,y);
nc1 = size(X(y==label(1)),1);
nc2 = size(X(y==label(2)),1);
l = [ones(nc1,1);-1*ones(nc2,1)];
L = double(bsxfun(@eq,l,l'));
tic
E = kMetricLearningMahalanobis(X,L,y,opts);
ti = toc;
fprintf('Estimating time matrix E: %.5f sec \n',ti)

for is = 1:length(Vs)
    
    fprintf('%i/%i\n',is,length(Vs));
    %% Training ETWSVM
    [Xc1,Xc2,labels] = SeparacionClases(X,y);

    [svmStructs] = TrainingETWSVM(...
     X,y,E,Vs(is),tipo);
  %% classification
    [label_estc,T] = VWLTSVM_Classify_CKA(svmStructs,TestX);

    svmPlus = svmStructs.svmPlus;
    svmMinus = svmStructs.svmMinus;
   %% Contourn over input space

    label_estc = reshape(label_estc,np+1,np+1);
    T1 = reshape(T(:,1),np+1,np+1);
    T2 = reshape(T(:,2),np+1,np+1);

    figure(1)
    [C,h] = contourf(Xm,Ym,T1,5,'LineColor','none');
    hold on
    plot(X(y==1,1),X(y==1,2),'r*'),
    plot(X(y==-1,1),X(y==-1,2),'b*'),
    colorbar
    legend({'Score Hyperplane +1';'+1';'-1'},'Location','Best')
    title(['Score hyperplane +1 sig=',num2str(Vs(is))]);

    figure(2)
    [C,h] = contourf(Xm,Ym,T2,5,'LineColor','none');
    hold on
    plot(X(y==1,1),X(y==1,2),'r*'),
    plot(X(y==-1,1),X(y==-1,2),'b*'),
    colorbar

    legend({'Score Hyperplane -1';'+1';'-1'},'Location','Best')
    title(['Score hyperplane -1 sig=',num2str(Vs(is))]);
    %% Graficas         

    figure(3)
    [C,h] = contourf(Xm,Ym,label_estc,40,'LineColor','none');
    hold on
    plot(X(y==1,1),X(y==1,2),'r*'),
    plot(X(y==-1,1),X(y==-1,2),'b*'),
    colorbar
    legend({'Hyperplane';'+1';'-1'},'Location','Best')
    title(['Classification sig=',num2str(Vs(is))]);

    figure(5)

    gscatter(X(:,1),X(:,2),y,'br','**')
    hold on
    plot(Xc1(svmMinus.alpha>1e-8,1),Xc1(svmMinus.alpha>1e-8,2),'go','LineWidth',2)
    plot(Xc2(-1*svmPlus.alpha>1e-8,1),Xc2(-1*svmPlus.alpha>1e-8,2),'ko','LineWidth',2)
    legend({'Class -1';'Class +1';'SV class -1';'SV class +1'},'Location','Best')
    title('Model support vectors')


    showfigs_c(2)

    fprintf('\t Press a key to continue... \n')
    pause
    hold off
    close all       

end
