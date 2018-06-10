function [struct]=TrainingETWSVM(X,t,E,sig,kernT)
% @copyright Cristian Alfonso Jimeneze -- craljimenez@utp.edu.co
%
%    Inputs:
%         X: input samples matrix NxP
%         t: labels column vector ti= 1 or -1
%         E: estimate matrix of CKA PxP'
%         sig: Gaussian kernel parameter
%         kernT: type kernel, lineal (linear) or noLineal (rbf) 
%    OUTPUTS:
%         struct : model struct 



    warning ('off','all');
    
    if nargin < 5
        kernT = 'Lineal';
           if nargin < 4
               sig = 1;
               if nargin < 3
                   E  = eye(size(Xc1,2));
                   if nargin < 2
                        error('Argumentos insuficientes')
                   end
               end
           end
    end
    
    kernT = lower(kernT);
%     %% Generacion matriz diagonales D1 y D2 para pesos 
%     X1s=Xc2;
%     Nneg2=size(Xc2,1);
%     Npos2=size(Xc1,1);
%     if (Npos2/Nneg2)<=1
%         D2=(Npos2/Nneg2)*eye(Nneg2);
%     elseif (Nneg2/Npos2)<1
%         D2=(Nneg2/Npos2)*eye(Nneg2);
%     end
%     Nneg1=size(X1s,1); % Tamaño del submuestreo para el hiperplano minoritario
%     if (Npos2/Nneg1)<=1
%         D1=(Npos2/Nneg1)*eye(Npos2);
%     elseif (Nneg1/Npos2)<1
%         D1=(Nneg1/Npos2)*eye(Npos2);
%     end
    %%
    [Xc1,Xc2,~] = SeparacionClases(X,t);
    
    S=[Xc1 ones(size(Xc1,1),1)]'; % Extend matrix, (q+1xN)
    Sp=[Xc2 ones(size(Xc2,1),1)]'; 
    d=size(S,1);
    
%     Ac1=((c1(1)*eye(d)+c2*(S*S'))*(Sp*Sp'));
%     Ac2=((c1(2)*eye(d)+c2*(Sp*Sp'))*(S*S'));
    invAc1=[E*E',ones(d-1,1);ones(1,d)];
    invAc2=invAc1;
    AA = [E;ones(1,size(E,2))];
%     Ac1=invAc1\eye(size(invAc1));
    if strcmp(kernT,'lineal')
        K1=(Sp'*Sp);
        K2=(S'*S);
        KAc1=Sp'*invAc1*Sp;
        KAc2=S'*invAc2*S;
    elseif strcmp(kernT,'nolineal')
        K1=exp(-pdist2(Sp',Sp').^2/(2*sig^2));
        K2=exp(-pdist2(S',S').^2/(2*sig^2));
%         K1=exp(-pdist2(X2s,X2s).^2/(2*sig^2))+1;
%         K2=exp(-pdist2(Xc1,Xc1).^2/(2*sig^2))+1;
        KAc1=exp(-pdist2(Sp'*AA,Sp'*AA));
        KAc2=exp(-pdist2(S'*AA,S'*AA));
%         KAc1=exp(-pdist2(Sp',Sp','mahalanobis',Ac1));
%         KAc2=exp(-pdist2(S',S','mahalanobis',Ac1));
%         KAc1=Sp'*invAc1*Sp;
%         KAc2=S'*invAc2*S;
          
    else
        error('Variable tipo no definida. Lineal o NoLineal')
    end
    H1=K1*KAc1;%+(D2\eye(size(D2)));  H1=min(H1,H1');
    H2=K2*KAc2;%+(D1\eye(size(D1))); H2=min(H2,H2');

    options = optimoptions('quadprog',...
    'Display','off');
    
    c2 = size(Xc1,1)/size(Xc2,1);
    
    %% QPP Hiper-plano mayoritario w_+ b_+
    alpha=quadprog(H1,-1*ones(size(Xc2,1),1),[],[],[],[],...
        zeros(size(Xc2,1),1),c2*ones(size(Xc2,1),1),[],options);
    %% QPP Hiper-plano minoritario w_- b_-
    gamma=quadprog(H2,-1*ones(size(Xc1,1),1),[],[],[],[],...
        zeros(size(Xc1,1),1),c2*ones(size(Xc1,1),1),[],options);

    %% Extraccion de w.
    Ac1=(Sp*Sp')*invAc1;
    Ac2=(S*S')*invAc2;
    z_Plus=-1*Ac1*Sp*alpha;
    z_Minus=Ac2*S*gamma;
    wPlus=z_Plus(1:end-1);
    wMinus=z_Minus(1:end-1);
    
    %%
    alphaClassify=-KAc1*alpha;
    gammaClassify=KAc2*gamma;
  
    
    svmPlus.alpha=-alpha;
    svmMinus.alpha=gamma;
    svmPlus.alphaClassify=alphaClassify;
    svmMinus.alphaClassify=gammaClassify;
    svmPlus.Sp=Sp;
    svmMinus.Sp=S;
    svmPlus.w=wPlus;
    svmMinus.w=wMinus;
    
    
    svmPlus.invA=invAc1;
    svmMinus.invA=invAc2;
%     svmPlus.K=K1;
%     svmMinus.K=K2;
    svmPlus.K_A=KAc1;
    svmMinus.K_A=KAc2;
%     svmPlus.w=wPlus;
%     svmMinus.w=wMinus;
    
    struct.svmPlus = svmPlus;
    struct.svmMinus = svmMinus;
    struct.sig = sig;
    struct.tipo = kernT;
    
    Xtestr = [Xc1;Xc2];
    nt=size(Xtestr,1);
    V=[Xtestr,ones(size(Xtestr,1),1)]';
    
    if strcmp(kernT,'lineal')
        T=[abs(V'*svmPlus.Sp*svmPlus.alphaClassify)/sqrt(svmPlus.w'*svmPlus.w),...
            abs(V'*svmMinus.Sp*svmMinus.alphaClassify)/sqrt(svmMinus.w'*svmMinus.w)];
    elseif strcmp(kernT,'nolineal')
        Kc1=exp(-pdist2(Xtestr,(svmPlus.Sp(1:end-1,:))').^2/(2*sig^2))+1;
        Kc2=exp(-pdist2(Xtestr,(svmMinus.Sp(1:end-1,:))').^2/(2*sig^2))+1;
        T=[abs(Kc1*svmPlus.alphaClassify)/sqrt(svmPlus.w'*svmPlus.w)...
            abs(Kc2*svmMinus.alphaClassify)/sqrt(svmMinus.w'*svmMinus.w)];
        %T=[abs(Kc1*svmPlus.alphaClassify)...
        %    abs(Kc2*svmMinus.alphaClassify)];
        
    end
    maxscore=max(T);
    svmPlus.maxs = maxscore(1);
    svmMinus.maxs = maxscore(2);
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    