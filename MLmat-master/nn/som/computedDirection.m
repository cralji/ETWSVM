% This function computes the update direction for neighbor Wk of the winner
% unit Wj regarding the input pattern Pj.
%
%     Dir = computedDirection(Wk, Pj, Pk, Nk, eta)
%
% The direction is selected to aim from Wk towards a point lying in the 
% plane formed by [Nk'  -Nk'*Pk]', where Pk is the closest pattern to Wk, 
% and Nk is the normal vector associated to the point Pk.
% This point results from the projection of a proportion of (Pj-Wk) upon
% the aforementioned plane.
%
% Written by Juan Bernardo G??mez Mendoza.
function Dir = computedDirection(Wk, Pj, Pk, Nk, eta)
    DV = Pj - Wk;   % Weight to point direction.
    if norm(DV)>0,
        DVn = DV/norm(DV);
    end;
    prop = eta*norm(DV);
    Wkd = Wk + prop*DVn;    % Weight displaced along DV.    
    WkdD = Nk'*(Wkd - Pk);  % Distance from Wk to the plane k.
    Wkdp = -WkdD*Nk + Wkd;   % Displaced weight, projected in plane k.
    Dir = Wkdp - Wk;    % Desired movement direction.
    if norm(Dir)>0,
        Dir = Dir/norm(Dir);
    end;
end