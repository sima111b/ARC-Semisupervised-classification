function [ hammingLoss]= binaryClassification_test(dataset)

folder='binary/';
restoredefaultpath;
addpath(genpath(pwd));
load(dataset);
%%Initialization **************************************************************************************
n_sample= size(groundTruth,2); %1000;
global n_node_features;
global n_nodes;
global n_pairs;

global weight_size_node;
global weight_size_pairwise;
n_nodes=size(groundTruth,2); % number of classes
n_node_features=size(feature_nodes,1);
n_pairs=(n_nodes*(n_nodes-1))/2;
weight_size_node=n_node_features;
weight_size_pairwise=n_node_features;

load('binary/theta_nodes_binary.mat');%
load('binary/theta_pairwise_binary.mat');
feature_pairwise=feature_pairwise_generator_binary(ones(n_nodes,1),feature_nodes,1); %81*81*300

%% Test
    lagrangianPotentials_node=(theta_nodes_binary*feature_nodes);
    lagrangianPotentials_pairwise=zeros(n_nodes,n_nodes);
    for slice = 1 : n_nodes
        for sl=1:n_nodes
            lagrangianPotentials_pairwise(slice,sl)=reshape(feature_pairwise(slice,sl,:),1,size(feature_pairwise,3))*theta_pairwise_binary; 
            
        end
        
    end
    lagrangianPotentials_pairwise=lagrangianPotentials_pairwise./n_pairs;
    
    [p_maximizer,game_value_maximizer,s_maximizer_binary]= DOMMulti_binary(groundTruth,theta_nodes_binary,feature_nodes,lagrangianPotentials_node,theta_pairwise_binary,feature_nodes,lagrangianPotentials_pairwise);
    
    [k,maxIndex]=max(p_maximizer);
    predicted_labels=s_maximizer_binary(maxIndex,:);
   hammingLoss=pdist2(groundTruth,predicted_labels,'hamming');
end

    