function [t_est,T]=PredictETWSVM(struct,Xtest)
    
    
    svmPlus = struct.svmPlus;
    svmMinus= struct.svmMinus;
    sig = struct.sig;
    tipo = lower(struct.tipo);
    
    Xtestr=Xtest;
    nt = size(Xtest,1);
    V=[Xtestr,ones(size(Xtest,1),1)]';
    
    if strcmp(tipo,'lineal')
        T=[abs(V'*svmPlus.Sp*svmPlus.alphaClassify),...
            abs(V'*svmMinus.Sp*svmMinus.alphaClassify)];
    elseif strcmp(tipo,'nolineal')
%         Kc1=exp(-pdist2(Xtest,(svmPlus.Sp(1:end-1,:))').^2/(2*sig^2))+1;
%         Kc2=exp(-pdist2(Xtest,(svmMinus.Sp(1:end-1,:))').^2/(2*sig^2))+1;
        Kc1=exp(-pdist2(V',svmPlus.Sp').^2/(2*sig^2));
        Kc2=exp(-pdist2(V',svmMinus.Sp').^2/(2*sig^2));
        T=[abs(Kc1*svmPlus.alphaClassify)...
            abs(Kc2*svmMinus.alphaClassify)];
        %T=[abs(Kc1*svmPlus.alphaClassify)...
        %    abs(Kc2*svmMinus.alphaClassify)];
        
    end
    T = T./repmat(max(T),nt,1);
    t_est = sign( T(:,2) - T(:,1) );
    t_est(t_est==0) = 1; %% seguridad que 
    %maxscore=max(T);
%     T(:,1) = T(:,1)/svmPlus.maxs;
%     T(:,2) = T(:,2)/svmMinus.maxs;
    %T(:,1)=T(:,1)./maxscore(1);
    %T(:,2)=T(:,2)./maxscore(2);
%     [~,IndexMin]=min(T,[],2);
%     [~,IndexMin]=max(T,[],2);
%     label_new=zeros(nt,1);
%     label_new(IndexMin==1)=1;
%     label_new(IndexMin==2)=-1;
%     label_new=sign(T(:,2)-T(:,1));
    