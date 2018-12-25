function [theta_nodes_binary,theta_pairwise_binary]= binaryClassification(dataset)
warning off;
clc;
clear;
folder='binary/';
restoredefaultpath;
addpath(genpath(pwd));
% profile -memory on;
% Please set the matrices in a way that the last dimention defines the
% number of entities in the matrix
load(dataset);
%%Initialization **************************************************************************************
save_after=20;
maxiteration=10000;
global n_nodes;
n_nodes=size(feature_nodes,2);
global n_pairs;
n_pairs=(n_nodes*(n_nodes-1))/2;
n_classes=2;

global weight_size_node;
global weight_size_pairwise;
weight_size_node=size(feature_nodes,1);

weight_size_pairwise=size(feature_nodes,1);
load('binary/theta_nodes_binary.mat');
theta_nodes_all=zeros(weight_size_node,maxiteration);
load('binary/theta_pairwise_binary.mat');
thetea_pairwise_all=zeros(weight_size_node,maxiteration);

feature_pairwise=feature_pairwise_generator_binary(ones(n_nodes,1),feature_nodes,1); 

game_values = zeros(maxiteration,1);  % the avg of game values over training examples
objective_values=zeros(maxiteration,1);

grads_magnitude_features = zeros(maxiteration,1); %
grads_magnitude_pairwise = zeros(maxiteration,1); %

grad_node= zeros (1,weight_size_node);  % % the sum of gradients over training examples in each batch
grad_pairwise = zeros (weight_size_pairwise,1);  % % the sum of gradients over training examples in each batch

avg_grad_batch_features= zeros (1,weight_size_node);
avg_grad_batch_pairwise = zeros (weight_size_pairwise,1);

% adagrad ********************************************
autocorr = 0.95;
fudge_factor=1e-6; %for numerical stability
master_stepsize = 1e-2;
historical_grad_nodes= zeros (1,weight_size_node);
historical_grad_pairwise= zeros (weight_size_pairwise,1);
%********************************************
%% Training
for itr = 1:maxiteration
            [grad_node,grad_pairwise,game_value_maximizer,objective_value_maximizer]=game_step_binary(feature_nodes,feature_pairwise,groundTruth,theta_nodes_binary,theta_pairwise_binary);
            grad_pairwise=reshape(sum(sum(grad_pairwise)),weight_size_pairwise,1)/n_pairs;

            %% adagrad 
            historical_grad_nodes=historical_grad_nodes+(grad_node.^2);
        historical_grad_pairwise=historical_grad_pairwise+(grad_pairwise.^2);
        adjusted_grad_nodes=grad_node./(sqrt(historical_grad_nodes)+fudge_factor);
        adjusted_grad_pairwise=grad_pairwise./(sqrt(historical_grad_pairwise)+fudge_factor);
        
        %% gradient update
        theta_nodes_binary=theta_nodes_binary - master_stepsize * adjusted_grad_nodes;
        theta_pairwise_binary=theta_pairwise_binary - master_stepsize * adjusted_grad_pairwise;
        theta_pairwise_binary=max(theta_pairwise_binary,0);
        %% Recording
         thetea_pairwise_all(:,itr)=theta_pairwise_binary;
         theta_nodes_all(:,itr)=theta_nodes_binary;

         game_values(itr) = game_value_maximizer;
         objective_values(itr)= objective_value_maximizer;
    if (objective_value_maximizer<0)
        itr
    end
        
    grads_magnitude_features(itr) = (1/(n_nodes*weight_size_node))*sum(sum(abs(grad_node)));
    grads_magnitude_pairwise(itr) = (1/weight_size_pairwise)*sum(abs(grad_pairwise));
    
    %%
       
    if (itr == maxiteration)
        
        disp('exceeded maximum iteration');
        
        break_condition = 'exceeded maximum iteration';
        
    end
    if( mod (itr, save_after) == 0 ) % based on data size, itr takes variable times. so save on update count instead
        
        %*****************************************************************
        
        fig=figure('Visible','off','Position', [0 0 1024 800]);
        
        plot(objective_values(1:itr));
                
        figName=strcat(folder,'Objectiveplot.png');
        
        saveas(fig, figName);
        %*****************************************************************
        
        fig=figure('Visible','off','Position', [0 0 1024 800]);
        
        plot(grads_magnitude_features(1:itr));
        figName=strcat(folder,'gradplot_node.png');        
        saveas(fig, figName);
        
        %*****************************************************************
        
        fig=figure('Visible','off','Position', [0 0 1024 800]);
        
        plot(grads_magnitude_pairwise(1:itr));
        figName=strcat(folder,'gradplot_pairwise.png');        
        saveas(fig, figName);
        %*****************************************************************
               fig=figure('Visible','off','Position', [0 0 1024 800]);
        
        plot(theta_nodes_all(:,1:itr)');
        figName=strcat(folder,'thetaplot_node.png');
        saveas(fig, figName);
        %*****************************************************************
        fig=figure('Visible','off','Position', [0 0 1024 800]);
        
        plot(thetea_pairwise_all(:,1:itr)');
        figName=strcat(folder,'thetaplot_pairwise.png');
        saveas(fig, figName);
        %*****************************************************************
               
        fig=figure('Visible','off','Position', [0 0 1024 800]);
        
        plot( game_values(1:itr));
        figName=strcat(folder,'game_values.png');
        saveas(fig, figName);
        %*****************************************************************

        save binary/theta_nodes_binary theta_nodes_binary;
        save binary/theta_pairwise_binary theta_pairwise_binary
    end
end
%% logging

fig=figure('Visible','off','Position', [0 0 1024 800]);

plot(grads_magnitude_features(1:itr));

saveas(fig,'finalgradplot_gpu.png');

save('lastrunallFeatures_gpu.mat');

fileID = fopen('output_gpu.txt','w');

fprintf(fileID, [break_condition '\n']);

