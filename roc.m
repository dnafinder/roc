function ROCout = roc(x,varargin)
%ROC Receiver Operating Characteristic (ROC) curve analysis.
%
%   Syntax
%   ------
%   ROCout = ROC(X)
%   ROCout = ROC(X, THRESHOLD)
%   ROCout = ROC(X, THRESHOLD, ALPHA)
%   ROCout = ROC(X, THRESHOLD, ALPHA, VERBOSE)
%   ROCout = ROC(X, THRESHOLD, ALPHA, VERBOSE, PLOTTING)
%   ROCout = ROC(X, THRESHOLD, ALPHA, VERBOSE, PLOTTING, PREVALENCE)
%
%   Description
%   -----------
%   ROC performs Receiver Operating Characteristic (ROC) curve analysis
%   for a continuous (or ordinal) diagnostic test or classifier.
%
%   The input matrix X is N-by-2:
%     * X(:,1) contains the test values
%     * X(:,2) contains the binary class labels:
%         - 1 = unhealthy / diseased
%         - 0 = healthy / non-diseased
%
%   For a set of candidate thresholds, the function computes:
%     * Sensitivity (True Positive Rate)
%     * Specificity (True Negative Rate)
%     * Positive and Negative Likelihood Ratios (PLR, NLR)
%     * An efficiency index (Matthews-like index)
%
%   The ROC curve is derived from Sensitivity and Specificity, and the
%   Area Under the Curve (AUC) is computed using the Mann–Whitney–
%   Wilcoxon U statistic (via MWWTEST). The standard error, confidence
%   interval for AUC, and a z-test for AUC > 0.5 are also provided.
%
%   When PREVALENCE is specified, the function also computes positive
%   and negative predictive values (PPV, NPV) at each threshold.
%
%   Inputs
%   ------
%   X          : N-by-2 numeric matrix.
%                X(:,1) = test values (real, finite, non-NaN, non-empty)
%                X(:,2) = class labels (0 = healthy, 1 = unhealthy).
%
%   THRESHOLD  : (optional) number of unique test values to be used
%                as candidate thresholds.
%                  * If THRESHOLD = 0 or omitted, all unique positive
%                    values in X(:,1) are used.
%                  * If THRESHOLD > 2, that number of quantile-based
%                    unique values is used (minimum allowed is 3).
%
%   ALPHA      : (optional) significance level for the AUC confidence
%                interval and z-test (default = 0.05; 0 < ALPHA < 1).
%
%   VERBOSE    : (optional) flag for textual output:
%                  0 = no textual report
%                  1 = print full report (default).
%
%   PLOTTING   : (optional) flag for graphical output:
%                  0 = no plots
%                  1 = produce ROC and cut-off plots (default).
%
%   PREVALENCE : (optional) true prevalence as probability in (0,1).
%                If provided, it is used to compute prior odds (POD)
%                and predictive values (PPV, NPV). If omitted or empty,
%                PPV/NPV and POD-based efficiency are not computed.
%
%   Outputs
%   -------
%   ROCout : structure with the following fields (if requested)
%
%     ROCout.AUC    : Area Under the ROC Curve.
%     ROCout.SE     : Standard error of the AUC.
%     ROCout.ci     : 1-ALPHA confidence interval for the AUC.
%     ROCout.p      : p-value for H0: AUC = 0.5 vs H1: AUC > 0.5.
%     ROCout.xr     : x-coordinates (False Positive Rate) of ROC points.
%     ROCout.yr     : y-coordinates (True Positive Rate) of ROC points.
%
%     If AUC is significantly greater than 0.5 and cut-off analysis
%     is performed:
%       ROCout.matrix : numeric table of cut-off data
%                       (cut-off, Se, Sp, efficiency, PLR, NLR,
%                        and optionally PPV, NPV when PREVALENCE is set).
%       ROCout.co     : selected cut-off points and related Se/Sp
%                       (max Se, max Sp, Se=Sp, max efficiency,
%                        max PLR, min NLR).
%       ROCout.table  : same as ROCout.matrix (for compatibility).
%
%   Example
%   -------
%   % X is an N-by-2 matrix: [test_value, class_label]
%   % class_label: 1 = diseased, 0 = healthy
%   X = [randn(50,1)+1,  ones(50,1); ...
%        randn(60,1),    zeros(60,1)];
%
%   ROCout = roc(X, 0, 0.05, 1, 1, 0.30);  % prevalence = 30%
%
%   Dependencies
%   ------------
%   This function requires:
%     * MWWTEST (Mann–Whitney–Wilcoxon test) available at:
%         https://github.com/dnafinder/mwwtest
%
%   Make sure MWWTEST.M is on the MATLAB path before calling ROC.
%
%   Notes
%   -----
%   * Class labels in X(:,2) must be 0 or 1. If all labels are 0
%     (only healthy) or all are 1 (only unhealthy), the function
%     throws an error.
%   * If the function is called without output argument and VERBOSE = 1,
%     results are displayed as formatted tables and comments in the
%     Command Window.
%   * The second plot (cut-off vs Se/Sp/Efficiency) uses nonlinear
%     fitting and smoothing splines from the Curve Fitting Toolbox.
%
%   References
%   ----------
%   Cardillo G. (2008). ROC curve: compute a Receiver Operating
%   Characteristics curve. MATLAB Central File Exchange.
%
%   ------------------------------------------------------------------
%   Author : Giuseppe Cardillo
%   Email  : giuseppe.cardillo.75@gmail.com
%   GitHub : https://github.com/dnafinder/roc
%   Created: 2008
%   Updated: 2025-11-26
%   Version: 2.0.0
%   ------------------------------------------------------------------

% --- Dependency check: MWWTEST must be available ------------------------
if exist('mwwtest','file') ~= 2
    error('ROC:MissingDependency', ...
        ['MWWTEST.m is required but was not found on the MATLAB path. ' ...
         'Please download it from https://github.com/dnafinder/mwwtest ' ...
         'and add it to your MATLAB path before calling ROC.']);
end

% Input Error handling
ip = inputParser;
addRequired(ip,'x',@(y) validateattributes(y,{'numeric'},...
    {'2d','real','finite','nonnan','nonempty','ncols',2}));
addOptional(ip,'threshold',0, @(y) isnumeric(y) && isreal(y) && isfinite(y) && ...
    isscalar(y) && (y==0 || y>2));
addOptional(ip,'alpha',0.05, @(y) validateattributes(y,{'numeric'},...
    {'scalar','real','finite','nonnan','>',0,'<',1}));
addOptional(ip,'verbose',1, @(y) isnumeric(y) && isreal(y) && isfinite(y) && ...
    isscalar(y) && (y==0 || y==1));
addOptional(ip,'plotting',1, @(y) isnumeric(y) && isreal(y) && isfinite(y) && ...
    isscalar(y) && (y==0 || y==1));
% prevalence as probability (0,1)
addOptional(ip,'prevalence',[], @(y) isempty(y) || ...
    (isnumeric(y) && isscalar(y) && isreal(y) && isfinite(y) && y>0 && y<1));

parse(ip,x,varargin{:});
threshold  = ip.Results.threshold;
alpha      = ip.Results.alpha;
verbose    = ip.Results.verbose;
plotting   = ip.Results.plotting;
prevalence = ip.Results.prevalence;
clear ip

assert(all(x(:,2)==0 | x(:,2)==1),'Warning: all x(:,2) values must be 0 or 1')
if all(x(:,2)==0)
    error('Warning: there are only healthy subjects!')
end
if all(x(:,2)==1)
    error('Warning: there are only unhealthy subjects!')
end

% Prevalence / prior odds (only via argument)
if ~isempty(prevalence)
    POD = prevalence/(1-prevalence); % prior odds
end

tr=repmat('-',1,100);

% Values to define thresholds (only positive, as in the original code)
z=sortrows(x,1);
z(z(:,1)<=0,:)=[];
if threshold==0
    labels=unique(z(:,1)); % all unique positive values
else
    K=linspace(0,1,threshold+1); K(1)=[];
    labels=quantile(unique(z(:,1)),K)';
end
clear z

ll=length(labels);       % number of thresholds
a = zeros(ll,2);         % Sensitivity & Specificity
c = zeros(ll,1);         % efficiency / Matthews-like index
if exist('POD','var')
    d = zeros(ll,2);     % PPV & NPV (if prevalence known)
end

ubar=median(x(x(:,2)==1),1); % unhealthy median value
hbar=median(x(x(:,2)==0),1); % healthy median value

% --- cumulative pre-computation for TP, FP, FN, TN ----------------------
[testValsSorted, sortIdx] = sort(x(:,1));
labelsSorted   = x(sortIdx,2);

isUnhealthy = (labelsSorted==1);
isHealthy   = (labelsSorted==0);

cumUnhealthy = cumsum(isUnhealthy);
cumHealthy   = cumsum(isHealthy);

nUnhealthy = cumUnhealthy(end);
nHealthy   = cumHealthy(end);
Nsorted    = numel(testValsSorted); %#ok<NASGU>

for K=1:ll
    thr = labels(K);
    if hbar<ubar
        % Case 1: higher values => more disease (original condition "value > thr")
        % Positive if value > thr
        idx = find(testValsSorted<=thr,1,'last'); % last index with value <= thr
        if isempty(idx)
            cumDidx = 0;
            cumHidx = 0;
        else
            cumDidx = cumUnhealthy(idx);
            cumHidx = cumHealthy(idx);
        end
        TP = nUnhealthy - cumDidx;
        FP = nHealthy   - cumHidx;
        FN = cumDidx;
        TN = cumHidx;
    else
        % Case 2: lower values => more disease (original condition "value < thr")
        % Positive if value < thr
        idx = find(testValsSorted<thr,1,'last'); % last index with value < thr
        if isempty(idx)
            cumDidx = 0;
            cumHidx = 0;
        else
            cumDidx = cumUnhealthy(idx);
            cumHidx = cumHealthy(idx);
        end
        TP = cumDidx;
        FP = cumHidx;
        FN = nUnhealthy - cumDidx;
        TN = nHealthy   - cumHidx;
    end

    M=[TP FP;FN TN];

    % Sensitivity and Specificity
    a(K,:)=diag(M)'./sum(M);

    % Matthews-like index / efficiency
    if exist('POD','var')
        PLR=a(K,1)/(1-a(K,2));
        NLR=(1-a(K,1))/(a(K,2));
        if ~isinf(PLR)
            PPV=1/(1+1/(PLR*POD));
            PPN=1/(1+NLR*POD);
            d(K,:)=[PPV PPN];
            J=sum(a(K,:))-1; PSI=PPV+PPN-1;
            if all([J PSI]) && all([J PSI]>0)
                c(K)=mean([1 geomean([J PSI])]);
            else
                c(K)=NaN;
            end
        else
            c(K)=NaN;
        end
    else
        % Matthews-like index from confusion matrix alone
        denom = sqrt(prod(sum(M,1))*prod(sum(M,2)));
        if denom>0 && isfinite(denom)
            matt = det(M)/denom;
            c(K)=mean([1 matt]);
        else
            c(K)=NaN;
        end
    end
end
clear K TP FP FN TN M cumDidx cumHidx

% Likelihood ratios
b=[a(:,1)./(1-a(:,2)) (1-a(:,1))./a(:,2)];

% ROC points
xroc=1-a(:,2); yroc=a(:,1);
if hbar>ubar
    xroc=flipud(xroc); yroc=flipud(yroc);
end

f=polyfit(1:1:length(yroc),yroc',1);
if sign(f(1))==-1
    if ~isequal([xroc(1) yroc(1)],[1 1])
        xroc=[1;xroc]; yroc=[1;yroc];
    end
    if ~isequal([xroc(end) yroc(end)],[0 0])
        xroc(end+1)=0; yroc(end+1)=0;
    end
elseif sign(f(1))==1
    if ~isequal([xroc(1) yroc(1)],[0 0])
        xroc=[0;xroc]; yroc=[0;yroc];
    end
    if ~isequal([xroc(end) yroc(end)],[1 1])
        xroc(end+1)=1; yroc(end+1)=1;
    end
end

% Use only the first column (test values) for MWWTEST
STATS=mwwtest(x(x(:,2)==1,1),x(x(:,2)==0,1));
% AUC from Mann-Whitney-Wilcoxon U statistics
Area=max(STATS.U)/prod(STATS.n);

% Standard error of AUC
lu=STATS.n(1); lh=STATS.n(2);
Area2=Area^2; Q1=Area/(2-Area); Q2=2*Area2/(1+Area);
V=(Area*(1-Area)+(lu-1)*(Q1-Area2)+(lh-1)*(Q2-Area2))/(lu*lh);
Serror=realsqrt(V);

% Confidence interval for AUC
ci=Area+[-1 1].*(realsqrt(2)*erfcinv(alpha)*Serror);
if ci(1)<0; ci(1)=0; end
if ci(2)>1; ci(2)=1; end

% z-test
SAUC=(Area-0.5)/Serror;
p=1-0.5*erfc(-SAUC/realsqrt(2));

if nargout
    ROCout.AUC=Area;
    ROCout.SE=Serror;
    ROCout.ci=ci;
    ROCout.p=p;
    ROCout.xr=xroc;
    ROCout.yr=yroc;
end

clear lu lh Area2 Q1 Q2 V

if verbose==1
    % Performance of the classifier
    if Area==1
        str='Perfect test';
    elseif Area>=0.90 && Area<1
        str='Excellent test';
    elseif Area>=0.80 && Area<0.90
        str='Good test';
    elseif Area>=0.70 && Area<0.80
        str='Fair test';
    elseif Area>=0.60 && Area<0.70
        str='Poor test';
    elseif Area>=0.50 && Area<0.60
        str='Fail test';
    else
        str='Failed test - less than chance';
    end

    % Display AUC results
    disp('ROC CURVE ANALYSIS')
    disp(tr)
    disp(cell2table({Area, Serror, ci, str, SAUC, p},...
        'VariableNames',{'AUC','Standard_error','Confidence_interval','Comment','Standard_AUC','p_value'}))
    if p<=alpha
        disp('The area is statistically greater than 0.5')
    else
        disp('The area is not statistically greater than 0.5')
    end
    disp(' ')

    if plotting==1
        % Display ROC curve
        H=figure;
        set(H,'Position',[4 402 560 420])
        axis square; hold on
        sh=stairs(xroc,yroc,'color','b','linewidth',2);
        fill([sh.XData(1),repelem(sh.XData(2:end),2)],...
             [repelem(sh.YData(1:end-1),2),sh.YData(end)],...
             'g','FaceAlpha',0.5)
        clear sh
        patch([0 1 1],[0 0 1],'r','FaceAlpha',0.5)
        set(gca,'Xtick',0:0.1:1)
        grid on
        hold off
        xlabel('False positive rate (1-Specificity)')
        ylabel('True positive rate (Sensitivity)')
        title(sprintf('ROC curve (AUC=%0.4f)',Area))
    end

    clear Area Serror ci str SAUC xroc yroc H

    if p<=alpha
        clear p
        if exist('POD','var')
            d((a(:,1)==0 & a(:,2)==1),1)=NaN;
            d((a(:,1)==1 & a(:,2)==0),2)=NaN;
            matrix=[labels'; a(:,1)'; a(:,2)';c';b(:,1)'; b(:,2)';d(:,1)'.*100; d(:,2)'.*100;]';
            clear POD a b c d labels
            if verbose
                disp('ROC CURVE DATA')
                disp(tr)
                disp(array2table(matrix,'VariableNames',{'Cut_off','Sensitivity','Specificity','Efficiency','PLR','NLR','Pos_pred','Neg_Pred'}))
            end
        else
            matrix=[labels'; a(:,1)'; a(:,2)';c';b(:,1)'; b(:,2)']';
            clear a b c d labels
            if verbose
                disp('ROC CURVE DATA')
                disp(tr)
                disp(array2table(matrix,'VariableNames',{'Cut_off','Sensitivity','Specificity','Efficiency','PLR','NLR'}))
            end
        end
        clear tr
        ROCout.matrix=matrix;

        if length(matrix(:,1))>2
            if hbar<ubar
                CSe=mean(matrix(matrix(:,1)==max(matrix(matrix(:,2)==max(matrix(:,2)),1)),1));% Max sensitivity cut-off
                CSp=mean(matrix(matrix(:,1)==min(matrix(matrix(:,3)==max(matrix(:,3)),1)),1));% Max specificity cut-off
            else
                CSe=mean(matrix(matrix(:,1)==min(matrix(matrix(:,2)==max(matrix(:,2)),1)),1));% Max sensitivity cut-off
                CSp=mean(matrix(matrix(:,1)==max(matrix(matrix(:,3)==max(matrix(:,3)),1)),1));% Max specificity cut-off
            end
            CEff=mean(min(matrix(matrix(:,4)==max(matrix(isfinite(matrix(:,4)),4)),1))); % Max efficiency cut-off
            CPlr=mean(min(matrix(matrix(:,5)==max(matrix(isfinite(matrix(:,5)),5)),1))); % Max PLR cut-off
            CNlr=mean(min(matrix(matrix(:,6)==min(matrix(isfinite(matrix(:,6)),6)),1))); % Min NLR cut-off

            % Shift cut-off values so that all (cutoff + COEFF) > 0 for nonlinear fitting.
            % The fit is performed on x + COEFF, but tick labels are shifted
            % back, so the curve is still expressed in terms of the original
            % cut-off values.
            minCut  = min(matrix(:,1));
            meanCut = mean(matrix(:,1));
            maxCut  = max(matrix(:,1));

            if minCut <= 0
                COEFF = 1 - minCut;
            else
                COEFF = 0;
            end

            mM = minCut  + COEFF;
            M  = meanCut + COEFF;
            MM = maxCut  + COEFF;

            ft = fittype( '1-1/((1+(x/C)^B)^E)', 'independent', 'x', 'dependent', 'y' );
            opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
            opts.Display = 'Off';
            opts.StartPoint = [0 M 0];
            if matrix(1,2)>matrix(end,2)
                opts.Lower = [-Inf mM 0];
                opts.Upper = [0 MM Inf];
            else
                opts.Lower = [0 mM 0];
                opts.Upper = [Inf MM Inf];
            end
            fitSe = fit(matrix(:,1)+COEFF,matrix(:,2), ft, opts );
            if matrix(1,3)>matrix(end,3)
                opts.Lower = [-Inf mM 0];
                opts.Upper = [0 MM Inf];
            else
                opts.Lower = [0 mM 0];
                opts.Upper = [Inf MM Inf];
            end
            fitSp = fit(matrix(:,1)+COEFF,matrix(:,3), ft, opts );

            % Fit efficiency vs cut-off using only finite values
            effIdx = isfinite(matrix(:,1)) & isfinite(matrix(:,4));
            xData  = matrix(effIdx,1) + COEFF;
            yData  = matrix(effIdx,4);

            [xData, yData] = prepareCurveData(xData, yData);
            ft     = fittype('smoothingspline');
            fitEff = fit(xData, yData, ft);

            clear xData yData ft

            myfun=@(x,se,sp) (-1./(1+(x./se(2)).^se(1)).^se(3))+(1./(1+(x./sp(2)).^sp(1)).^sp(3));
            SeSp=fzero(@(x) myfun(x,coeffvalues(fitSe),coeffvalues(fitSp)),M);

            clear mM M MM ft opts myfun

            if plotting==1
                xg=linspace(0,max(matrix(:,1))+COEFF,500);
                H2=figure;
                set(H2,'Position',[570 402 868 420])
                hold on
                H=ones(1,9);
                c=[0 0 1;1 0 0; 0 1 0; 0 0 0.1724; 1 0.1034 0.7241; 1 0.8276 0];
                H(1) = plot(xg,feval(fitSe,xg), 'marker','none','linestyle','-', 'color',c(2,:), 'linewidth',2);
                H(2) = plot([CSe CSe]+COEFF,[0 1], 'marker','none','linestyle','--','color',c(2,:), 'linewidth',2);
                H(3) = plot(xg,feval(fitSp,xg), 'marker','none','linestyle','-', 'color',c(3,:), 'linewidth',2);
                H(4) = plot([CSp CSp]+COEFF,[0 1], 'marker','none','linestyle','--','color',c(3,:), 'linewidth',2);
                H(5) = plot(xg,feval(fitEff,xg),'marker','none','linestyle','-', 'color',c(1,:), 'linewidth',2);
                H(6) = plot([CEff CEff]+COEFF,[0 1],'marker','none','linestyle','--','color',c(1,:), 'linewidth',2);
                H(7) = plot([SeSp SeSp],[0 1],'marker','none','linestyle','--','color',c(6,:), 'linewidth',2);
                H(8) = plot([CPlr CPlr]+COEFF,[0 1],'marker','none','linestyle','--','color',c(5,:), 'linewidth',2);
                H(9) = plot([CNlr CNlr]+COEFF,[0 1],'marker','none','linestyle','--','color',c(4,:), 'linewidth',2);
                xlabel('Test cut-off')
                ylabel('Percent')
                hold off
                legend(H,...
                    'Sensitivity',sprintf('Max Sensitivity cutoff: %0.4f',CSe),...
                    'Specificity',sprintf('Max Specificity cutoff: %0.4f',CSp),...
                    'Efficiency',sprintf('Max Efficiency cutoff: %0.4f',CEff),...
                    sprintf('Cost Effective cutoff: %0.4f',SeSp-COEFF),...
                    sprintf('Max PLR: %0.4f',CPlr),sprintf('Min NLR: %0.4f',CNlr),...
                    'Location','BestOutside')
                axis([xg(1) xg(end) 0 1.1])

                % Shift tick labels back to original cut-off scale
                if COEFF~=0
                    xt  = get(gca,'XTick');
                    xtl = arrayfun(@(v) sprintf('%0.2f',v-COEFF), xt, 'UniformOutput', false);
                    set(gca,'XTick',xt,'XTickLabel',xtl)
                end
            end

            z=fitSe(SeSp-COEFF);
            fprintf('1) Max Sensitivity Cut-off point= %0.4f\n',CSe)
            fprintf('2) Max Specificity Cut-off point= %0.4f\n',CSp)
            fprintf('3) Cost effective Cut-off point (Sensitivity=Specificity=%0.4f)= %0.4f\n',z,SeSp-COEFF)
            fprintf('4) Max Efficiency Cut-off point= %0.4f\n',CEff)
            fprintf('5) Max PLR Cut-off point= %0.4f\n',CPlr)
            fprintf('6) Min NLR Cut-off point= %0.4f\n',CNlr)
            m=[CSe matrix(matrix(:,1)==CSe,2:3); CSp matrix(matrix(:,1)==CSp,2:3);...
                SeSp-COEFF z z; CEff matrix(matrix(:,1)==CEff,2:3);...
                CPlr matrix(matrix(:,1)==CPlr,2:3); CNlr matrix(matrix(:,1)==CNlr,2:3);];
        end
    else
        m=NaN;
        matrix=NaN;
    end

    if nargout
        ROCout.co=m;     % cut-off points
        ROCout.table=matrix;
    end
end
