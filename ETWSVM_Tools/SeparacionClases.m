function [Xc1,Xc2,labels]=SeparacionClases(XTRAIN,tTRAIN)
%%%[Xc1,Xc2,labels]=SeparacionClases(XTRAIN,tTRAIN)
%%% Xc1: Minoritory samples
%%% Xc2: Majoritory samples
%%% labels : target, the firs is  the minoritory class and other the
%%%          majoritory class

    if nargin<2
        error('Faltan argumentos necesarios en la funcion. \n \t [Xc1,Xc2,labels]=SeparacionClases(XTRAIN,tTRAIN)')
    end

    labels=unique(tTRAIN);
    nClass=length(labels);

    if (nClass>2) && (nClass<2)
        error('Existen mas de dos clases. Solo se hace clasificación binaria')
    end
    if nClass==1
        error('Solo una clase');
    end
    %% Determina la clase minoritaria y mayoritaria
    
    
    if iscell(tTRAIN)
        B = [numel(find(char(tTRAIN)==labels{1})) numel(find(char(tTRAIN)==labels{2}))];
        [~,indmin] = min(B);
        [~,indmax] = max(B);
        Xc1=XTRAIN(labels{indmin}==char(tTRAIN),:);
        Xc2=XTRAIN(labels{indmax}==char(tTRAIN),:);
        labels = {labels{indmin};labels{indmax}};
    else
        B = [numel(find(tTRAIN==labels(1))) numel(find(tTRAIN==labels(2)))];
        [~,indmin] = min(B);
        [~,indmax] = max(B);
        Xc1=XTRAIN(labels(indmin)==tTRAIN,:);
        Xc2=XTRAIN(labels(indmax)==tTRAIN,:);
        labels = [labels(indmin);labels(indmax)];
    end