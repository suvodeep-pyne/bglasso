% Matlab program to implement slice sampler for Dirichlet process mixture model and implement 
% for data simulated in previous analysis under lpp
% 1-30-2008

% --- define global constants --- %
%load lpp1out.mat
gtot = 1000; %15000; 
cur = 1; 
nrun = 1000; % 15000; 
burn = 2500; 
thin = 1; 
km = 100;      % max number of global species (shouldn't come close to this max)
count = 100;

% p is the dimensions of the matrix
p = 20;

% Sample size: no of subjects
n = 10;

lambdapath = [linspace(0,0.005,100) linspace(0.0052,0.05,200) linspace(0.05,0.5,100)]; ll = length(lambdapath);
threshpath = [0.3:0.01:0.7]; ll1 = length(threshpath);

count = 50; 
mip = linspace(0,0.995,100); ll1 = length(mip);
rhopath = linspace(0.005,0.3,10); RR = length(rhopath);

TP_rj = zeros(count,ll1,RR); FP_rj = zeros(count,ll1,RR); TN_rj = zeros(count,ll1,RR); FN_rj = zeros(count,ll1,RR);
spec_rj = zeros(count,ll1,RR); sens_rj = zeros(count,ll1,RR); MCC_rj = zeros(count,ll1,RR); precision_rj = zeros(count,ll1,RR);
spec20_rj = zeros(count,ll1); sens20_rj = zeros(count,ll1);
TP20_rj = zeros(count,ll1); FP20_rj = zeros(count,ll1); TN20_rj = zeros(count,ll1); FN20_rj = zeros(count,ll1);
ROC_rj = zeros(count,1,RR); PRC_rj = zeros(count,1,RR); accrate1 = zeros(count,1); ROC20_rj = zeros(count,1);
p_rj_est = zeros(count, p*(p+1)/2 - p);
q = p;
burnin = 10000; nmc = 90000;
b0=10; D0= .8*eye(q);


for lp= 26:count
if (lp<=25)
n = 100; p = 300; p1=p/6;
elseif (lp>25)
n=100; p=200; p1 = p/4;
end

%%%% AR(1) case
SigmaT = toeplitz(0.7.^[0:p-1]);
OmegaT = inv(SigmaT);

%%%% AR(2) case 
%OmegaT = toeplitz([1,0.5,0.25,zeros(1,p-3)]);
%SigmaT = inv(OmegaT);


%%%% Block case
%SigmaT = eye(p);
%SigmaT(1:p/2,1:p/2) = 0.5*ones(p/2)+(1-0.5)*eye(p/2);
%SigmaT(p/2+1:end,p/2+1:end) = 0.5*ones(p/2)+(1-0.5)*eye(p/2);
%OmegaT = inv(SigmaT);


%%% Star case
%OmegaT = eye(p); OmegaT(1,2:end) = 0.1; OmegaT(2:end,1) = 0.1;
%SigmaT = inv(OmegaT);


%%% Circle case
%SigmaT = inv(toeplitz([2,1,zeros(1,p-3),0.9]));
%OmegaT = inv(SigmaT); 

%%% Full case
%OmegaT = ones(p)+eye(p);
%SigmaT = inv(OmegaT);

%%% Long range dependence
%SigmaT = zeros(p,p); 
%for i=1:p
%    ind = i:p;
%     SigmaT(i,ind) = 0.5*(abs(abs(i-ind) + 1).^(1.4) - 2*abs(i - ind).^(1.4) + abs(abs(i-ind) - 1).^(1.4));
%     SigmaT(ind,i) = SigmaT(i,ind);
%end
%OmegaT = inv(SigmaT);

%%%%%%%%%%%% Data Generation %%%%%%%%%%%%%%%%%%%%
Y = zeros(n,p);
for ii=1:n
    Y(ii,:)  = [-ones(1,p1) ones(1,p1) 2.5*ones(1,p1) -2.5*ones(1,p1)] + mvnrnd(zeros(1,p),SigmaT);
end
 YY = Y;
 Y = (Y - repmat(mean(Y),[n 1]))./repmat(sqrt(var(Y)),[n 1]);
 Yt = (Y - repmat(mean(Y),[n 1]))./repmat(sqrt(var(Y)),[n 1]); 
 S = Yt'*Yt;
 
if cur == 1
   % --- Define hyperparameter values --- %
   b = 3; % Degrees of freedom for inverse Wishart
   aa = 1; ba = 1;     % gamma hyperparameters for alpha
   ap = 1000000; bp = 1;     % gamma hyperparameters for residual precision, psi
   bl = 1; al = p;
   tht0 = zeros(n,1);  % prior mean for base measure 
   Ptht0 = 0.01*eye(n);
   
   % --- Initial values --- %
   a = 1;                             % DP precision parameter
   ps = 10;                           % residual precision 
   Tht = mvnrnd(zeros(1,n),eye(n),[km 1]);                 % initial atom values
   ph = kmeans(YY',3,'replicates',100);  ph1 = ph; phk = ph;    % initial atoms allocation index 
   nn = 50*ones(p/50,1);
   nus = betarnd(1,a,[1 km]);         % stick-breaking random variables
   nu = nus.*cumprod([1 1-nus(:,1:km-1)]); % category probabilities
   tht = zeros(n,p);                  % atoms for each subject
   u = unifrnd(0,nu(ph)');
   kjs = max(ph);    
   lambda = 10;   
   D0 = gamrnd(lambda.^2/2,10);
   alpha = mvnrnd(zeros(1,p),eye(p));
   Omega = wishrnd(inv(diag(D0)),b+1);
   ADJ = eye(p);
   
   % --- Define output files --- %
   kjsout = zeros(gtot/thin,1);         % max occupied atom index for each component
   phout = zeros(gtot/thin,p);
   thtout = zeros(gtot/thin,n,p);       % thts for each subject	
   Dout = zeros(gtot/thin,1);        % vector of diagonals for D matrix
   lambdaout = zeros(gtot/thin,1);   % gamma hyperparameters for diagonals of D^{-1}
   Omegaout = zeros(gtot/thin,p*(p+1)/2);  % vectorized upper triangular Omega with p(p+1)/2 elements
   precout = zeros(gtot/thin,p); %vector of diagonals of Omega
   psout = zeros(gtot/thin,1);   %residual precision
   alphaout = zeros(gtot/thin,p);		  	
   nnout = zeros(gtot/thin,km);  
end

% -- Gibbs sampling steps -- %
for g = cur:(cur+nrun)

   b = 3; % Degrees of freedom for Wishart distribution
   for h=1:kjs+5
	  nus(h) = betarnd(1 + sum(ph==h), a + sum(ph>h))';
      nu(h) = nus(h)*prod([1 1-nus(1:h-1)]);
	  if sum(nu(1:h)) > 1-min(u), break, end
    end
    
    u = unifrnd(0,nu(ph));          
    
   	% -- update allocation to atoms -- %
	pih = -10000000*ones(p,h);       % probabilities of allocation to each atom for each subject 
	R = zeros(p,h);             % indexes which atoms are available to each subject (depends on their u)
	for l = 1:h 
	       ind = u<nu(l); 
		   R(ind,l)=1;      
		   pih(ind,l) = -(ps/2)*sum((Y(:,ind) - repmat(alpha(ind),[n 1]) - repmat(Tht(l,:)',[1 sum(ind)])).^2,1); 
    end
    pih = R.*exp(pih - repmat(max(pih')',[1 h])); 
	pih = pih./repmat(pih*ones(h,1),[1 h]);                  % normalize
	pih = [zeros(p,1) cumsum(pih')'];
	r = unifrnd(0,1,[p,1]); 
    
    for l = 1:h
		ind = (r>pih(:,l) )&(r<=pih(:,l+1) );         
		ph(ind) = l;         
		tht(:,ind) = repmat(Tht(l,:)',[1 sum(ind)]);
	end

    % -- Update atoms -- %
    for h = 1:max(ph)
      if (sum(ph==h)>0)
      Vtht = inv(Ptht0 + ps*sum(ph==h)*eye(n) ); 
	  A = chol(Vtht); Vtht = A'*A; 
	  Etht = Vtht*(Ptht0*tht0 + ps*sum(Y(:,ph==h) - repmat(alpha(ph==h),[n 1]),2)); 
      Tht(h,:) = mvnrnd(Etht,Vtht);
	  tht(:,ph==h) = repmat(Tht(h,:)',[1 sum(ph==h)]);
      end    
    end 
    
    
    Ptht0 = 0.01*eye(n);
    % -- Update residual precision psi -- %
    ps = gamrnd(ap + (n*p)/2,1./(bp + 0.5*sum(sum((Y-tht-repmat(alpha,[n 1])).^2))));
    % ps = 0.01;

    % -- Update a -- %
    const = 0;
    kjs = max(ph); 
    a = gamrnd(aa + length(unique(ph)),1./(ba - sum(log(1-nus(1:kjs)))));     
  
    nn = zeros(max(ph),1); 
    for kk=1:max(ph)
        nn(kk) = sum(ph==kk);   
        iD = inv(D0*eye(nn(kk)) + (Y(:,ph==kk)-tht(:,ph==kk))'*(Y(:,ph==kk)-tht(:,ph==kk)) );
        iD = (iD + iD')/2;
        SSY = mvnrnd(zeros(b+n, nn(kk)),iD);
        Omega1 = SSY'*SSY;  
        Omega(ph==kk,ph==kk) = Omega1;
        Valpha1 = inv(Omega1 + n*ps*eye(nn(kk)));
        Ealpha1 = Valpha1*(ps.*sum(Y(:,ph==kk)-tht(:,ph==kk))');
        alpha(ph==kk) = mvnrnd(Ealpha1,Valpha1); 
    end
    
    Omega = (Omega + Omega')/2;
    prec = diag(Omega); 
    D0 = 10;
    
    %mu = lambda'/sum(diag(Omega));
    %D0 = rand_ig(mu',lambda.^2);
    %bl1 = bl + sqrt(sum(diag(Omega)))/2;
    %al1 = al + b*p/2;
    %lambda = gamrnd(al1,1./bl1)'; 
        
    dummy = Omega(1,1:p);
    for j=2:p
        dummy = [dummy Omega(j,j:p)];
    end
       
    % -- save sampled values (after thinning) -- %
    if mod(g,thin)==0
       psout(g/thin,:) = ps;
       kjsout(g/thin) = kjs;
	   alphaout(g/thin,:) = alpha; 
	   thtout(g/thin,:,:) = tht; 
       phout(g/thin,:) = ph';
       Omegaout(g/thin,:) = dummy;
       precout(g/thin,:) = prec';
       Dout(g/thin,:) = D0;
       lambdaout(g/thin,:) = lambda';
       nnout(g/thin,1:length(nn)) = nn';
     end
    if(abs(g/100-round(g/100))==0)
    [lp g ps nn' mean(mean(abs(tht+repmat(alpha,[n 1]) - YY ))) ]
    end
end
PrecT = diag(OmegaT);
rhoT = 2*eye(p) -OmegaT./sqrt(diag(OmegaT)*diag(OmegaT)');
dummyT = OmegaT(1,1:p);
    for j=2:p
        dummyT = [dummyT OmegaT(j,j:p)];
    end

pi_est = zeros(p,p);
for ii=1:p
  for jj=(ii+1):p
       pi_est(ii,jj) = mean(phout(5000:g,ii)==phout(5000:g,jj));
       pi_est(jj,ii) = pi_est(ii,jj);
  end
end

dist = zeros(g-5000,1);
for kk=5001:g
    ed = 0;
    for ii=1:p
        AA = (phout(kk,ii)==phout(kk,:));
        ed = ed + sum(abs(AA-pi_est(ii,:)));
    end
    dist(kk-5000) = ed;
end
ind = 5000 + find(dist==min(dist),1,'first'); % index of optimal clustering using Dahl (2006) method
phind = phout(ind,:);

tht_est = mean(thtout(5000+find(dist==min(dist)),:,:),1);
tht_est = reshape(tht_est, [n p]);
nn_est = zeros(length(unique(phind)),1);
for kk=1:max(phind)
    nn_est(kk) = sum(phind==kk);
end

D_est = mean(Dout(5000:g,:),1);
prec_est = mean(precout(5000:g,:));
Omega_est = zeros(p,p); OmegaL = zeros(p,p); OmegaU = zeros(p,p);
for ii=1:p
    t1 = ((ii-1)*p+1- max(0,sum(1:ii-2)));
    t2 = (ii-1)*p+1 -max(0,sum(1:ii-2))+ p- ii ;
    Omega_est(ii,ii:p) = mean(Omegaout(5000:g,t1:t2),1); 
    Omega_est(ii:p,ii) = Omega_est(ii,ii:p);
    OmegaL(ii,ii:p) = quantile(Omegaout(5000:g,t1:t2),0.025,1);
    OmegaL(ii:p,ii) = OmegaL(ii,ii:p);
    OmegaU(ii,ii:p) = quantile(Omegaout(5000:g,t1:t2),0.975,1);
    OmegaU(ii:p,ii) = OmegaU(ii,ii:p);    
end
rho_est = 2*eye(p) - Omega_est./sqrt(diag(Omega_est)*diag(Omega_est)');
% Drton & Perlman's method: Does not work in our case when n<p
%z_score = 0.5*log((1+rho_est)/(1-rho_est));
%cutoff = norminv(0.5*(0.95)^(2/(p*(p-1)))+0.5, 0 ,1);
%adj_init = (z_score - cutoff/sqrt(n-p) >0)||(z_score + cutoff/sqrt(n-p)<0);

adj_init = (max(OmegaL,OmegaU)<0)|(min(OmegaL,OmegaU)>0);

%%%%%%%%% Post MCMC graph determination based on hyper inverse Wishart %%%%
% Uses Matlab parallel computing
burnin = 10000; nmc = 100000; 
pADJ_est = zeros(length(nn_est),p,p);

for kk=1:max(phind)
   D0_est = mean(Dout(5000+ find(dist == min(dist))))*eye(nn_est(kk));
   if(nn_est(kk)>1)
   pMatdum = eye(p);
   %adj_o = eye(nn_est(kk));
   adj_o = adj_init(ph==kk,ph==kk);
   [adj_save] = SSUR_swapGraph(Y(:,phind==kk)-tht_est(:,phind==kk),b,D0_est,burnin,nmc,adj_o,n,nn_est(kk)); %dim(adj_save) = nmc X nn_est(kk) X nn_est(kk)
   pMatdum(1:nn_est(kk),1:nn_est(kk)) = mean(adj_save,3);
   end
   pADJ_est(kk,:,:) = pMatdum;   
end   

pADJ_est1 = eye(p); 
for kk=1:max(phind)
     pADJ_est1(phind==kk, phind==kk) = pADJ_est(kk,1:nn_est(kk),1:nn_est(kk));  
end
   

for rr=1:length(rhopath)

TN_rj(lp,:,rr)=0; TP_rj(lp,:,rr) = 0; FP_rj(lp,:,rr) = 0; FN_rj(lp,:,rr) = 0; TP20_rj(lp,:) = 0; TN20_rj(lp,:) = 0; FP20_rj(lp,:) = 0;FN20_rj(lp,:) = 0;

for ii=1:ll1
for j=1:p
    ind = 1:p; ind(j) = [];
    TN_rj(lp,ii,rr) = TN_rj(lp,ii,rr) + sum( (pADJ_est1(j,ind)< mip(ii) ).*(abs(OmegaT(j,ind)) < rhopath(rr)) );
    TP_rj(lp,ii,rr) = TP_rj(lp,ii,rr) + sum( (pADJ_est1(j,ind)>=mip(ii) ).*(abs(OmegaT(j,ind)) > rhopath(rr)) );
    FP_rj(lp,ii,rr) = FP_rj(lp,ii,rr) + sum( (pADJ_est1(j,ind)>=mip(ii) ).*(abs(OmegaT(j,ind)) < rhopath(rr)) );
    FN_rj(lp,ii,rr) = FN_rj(lp,ii,rr) + sum( (pADJ_est1(j,ind)<mip(ii) ).*(abs(OmegaT(j,ind)) > rhopath(rr)) );
    TN20_rj(lp,ii) = TN20_rj(lp,ii) + sum( (pADJ_est1(j,ind)< mip(ii) ).*(abs(OmegaT(j,ind)) <=0.1) );
    TP20_rj(lp,ii) = TP20_rj(lp,ii) + sum( (pADJ_est1(j,ind)>=mip(ii) ).*(abs(OmegaT(j,ind)) >0.1) );
    FP20_rj(lp,ii) = FP20_rj(lp,ii) + sum( (pADJ_est1(j,ind)>=mip(ii) ).*(abs(OmegaT(j,ind)) <=0.1) );
    FN20_rj(lp,ii) = FN20_rj(lp,ii) + sum( (pADJ_est1(j,ind)<mip(ii) ).*(abs(OmegaT(j,ind)) >0.1) );
end
spec_rj(lp,ii,rr) = TN_rj(lp,ii,rr)/(TN_rj(lp,ii,rr)+FP_rj(lp,ii,rr)); sens_rj(lp,ii,rr) = TP_rj(lp,ii,rr)/(TP_rj(lp,ii,rr)+FN_rj(lp,ii,rr));
%MCC_rj(lp,ii) = (TP_rj(lp,ii)*TN_rj(lp,ii) - FP_rj(lp,ii)*FN_rj(lp,ii))/sqrt((TP_rj(lp,ii)+FP_rj(lp,ii))*(TP_rj(lp,ii)+FN_rj(lp,ii))*(TN_rj(lp,ii)+FP_rj(lp,ii))*(TN_rj(lp,ii)+FN_rj(lp,ii)));
spec20_rj(lp,ii) = TN20_rj(lp,ii)/(TN20_rj(lp,ii)+FP20_rj(lp,ii)); sens20_rj(lp,ii) = TP20_rj(lp,ii)/(TP20_rj(lp,ii)+FN20_rj(lp,ii));
%TP10_rj(lp,ii) = TP10_rj(lp,ii)/(sum(sum(abs(rhoT)>0.1))-p);TP20_rj(lp,ii) = TP20_rj(lp,ii)/(sum(sum(abs(rhoT)>0.1))-p);TP30_rj(lp,ii) = TP30_rj(lp,ii)/(sum(sum(abs(rhoT)>0.3))-p);TP40_rj(lp,ii) = TP40_rj(lp,ii)/(sum(sum(abs(rhoT)>0.4))-p);
end
spec1 = 1-spec_rj(lp,:,rr);  
ROC_rj(lp,rr) =  polyarea([1-spec_rj(lp,:,rr) 1-spec_rj(lp,length(spec1),rr) 1],[sens_rj(lp,:,rr) 0 0]);
TE = (TP_rj(lp,:,rr) + FP_rj(lp,:,rr));
precision_rj(lp,(TE>0),rr) = TP_rj(lp,(TE>0),rr)./TE(TE>0);
precision1 = precision_rj(lp,(TE>0),rr);
PRC_rj(lp,rr) = polyarea([sens_rj(lp,(TE>0),rr) sens_rj(lp,length(TE>0),rr) sens_rj(lp,1,rr)], [precision1 0 0]);
%PRC1(lp,rr) = polyarea([1-spec(lp,:,rr) 1-spec(lp,ll,rr) 1-spec(lp,1,rr) ], [precision1 0 0 ]);
spec1 = 1-spec20_rj(lp,:);  
ROC20_rj(lp) =  polyarea([1-spec20_rj(lp,:) 1-spec20_rj(lp,length(spec1)) 1],[sens20_rj(lp,:) 0 0]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


end  %% End of lp Loop



