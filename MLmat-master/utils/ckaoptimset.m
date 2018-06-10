function options = ckaoptimset(options_in)
% options_out = ckaoptimset(options_in)
% Creates CKA option structure with fields:
%   Q \in R^+                 : Output dimension criterion (default = 0). 
%                               See ml_pca.
%   showWindows               : If true every 10 iterations the gradient
%                               based results are shown (default = true).
%   showCommandLine           : If true every 10 iterations the gradient
%                               based results are printed (default = true).
%   lr \in R^2                : Maximum and minimum learning rates 
%                               (default = [1e-4 1e-5]).
%   goal \in Real             : Performance goal (default = -Inf).
%   maxiter                   : Maximum number of iterations (default =
%                               10*numberOfVariables).
%   min_grad \in R^+          : Minimum gradient (default = 1e-5).
%   init                      : Initizalization method: 'pca' (default)
%                               Principal Components Analysis or P x Q 
%                               initial matrix.
%   training \in bool^{N x 1} : Training indexes (default = true^{N x 1}).
%   max_fail \in Z^+          : Maximum validation checks (default = 10);
options = struct('showWindow',true,...
                 'showCommandLine',true,...
                 'lr',[1e-4 1e-5],...
                 'min_grad',1e-5,...
                 'goal',-Inf,...
                 'maxiter','default',...
                 'training','default',...
                 'Q',0,...
                 'init','pca',...
                 'max_fail',10);

if nargin>0
  fields = fieldnames(options_in);
  for f = fields'
    options.(f{1}) = options_in.(f{1});
  end
end