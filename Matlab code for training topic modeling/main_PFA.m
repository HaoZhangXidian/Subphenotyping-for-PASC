clc
clear
close all


load('../dataset/COVID_positive_any_new_PASC_Feature_DX.mat')

X = Feature_DX';
clear Feature_DX

%% parameter
K = 10;

V= size(X,1);
N = size(X,2);

iteration = 1000;
step=1;

%% hyper-parameter
Supara.eta = 0.01;
Supara.epilson0 = 0.1;
%% Initialization
Phi = rand(V,K);
Phi = bsxfun(@rdivide, Phi, max(realmin,sum(Phi,1)));
Phi_ave = 0;

r = 0.01*ones(K,1);
Theta    = ones(K,N)/K;
Theta_ave = 0;

Likelihood = zeros(iteration,1);
flag = 0;
burn_in = 500;

%% Learn
for i=1:iteration    
    
    
    %% sample latent count
    [x_dotkn,x_vkdot] = Multrnd_Matrix_mex_fast_v1(sparse(X), Phi, Theta);
    
    %% sample phi
    Phi = SamplePhi(x_vkdot,Supara.eta);
    if nnz(isnan(Phi))
        warning('Diagnosis Phi Nan');
        Phi(isnan(Phi)) = 0;
    end
    
    %% sample theta
    shape = bsxfun(@plus,r,x_dotkn);
    scale = 1;
    Theta = bsxfun(@times, randg(shape),  scale);
    
    %%
    if (i>burn_in) && (mod(i,step)==0)
        Theta_ave = Theta_ave + Theta;
        Phi_ave = Phi_ave + Phi;
        flag = flag + 1;
        Theta_mean = Theta_ave/flag;
        Phi_mean = Phi_ave/flag;
        
    end    
    
    
    
    %% Calculate likelihood
    Lambda = Phi*Theta;
    likelihood = sum(sum(full(X) .* log(Lambda)-Lambda))/V;
    
    Likelihood(i) = likelihood;
    
    if (i>burn_in) && (mod(i,step)==0)
        Lambda = Phi_mean*Theta_mean;
        likelihood_mean = sum(sum(full(X) .* log(Lambda)-Lambda))/V;
    end
    
    if (i>burn_in) && (mod(i,step)==0)
        fprintf('Iteration %d/%d, Likelihood %f, Likelihood mean %f\n',i, iteration, likelihood, likelihood_mean);
    else
        fprintf('Iteration %d/%d, Likelihood %f \n',i, iteration, likelihood);
    end
end

save('..\trained_topic_model\PFA_trained_model.mat','Phi_mean','Theta_mean')

