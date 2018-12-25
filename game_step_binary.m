function [sample_grad_node,sample_grad_pairwise,game_value_maximizer,objective_value_maximizer]=...
    game_step_binary(feature_nodes,feature_pairwise,groundTruth,theta_nodes,theta_pairwise)
global n_pairs;
global n_nodes;

global weight_size_node;
global weight_size_pairwise;
lagrangianPotentials_node=(theta_nodes*feature_nodes);

lagrangianPotentials_pairwise=zeros(n_nodes,n_nodes);
for slice = 1 :n_nodes
    for sl=1:n_nodes
        lagrangianPotentials_pairwise(slice,sl)=reshape(feature_pairwise(slice,sl,:),1,size(feature_pairwise,3))*theta_pairwise; 
    end
end
lagrangianPotentials_pairwise=lagrangianPotentials_pairwise./n_pairs;
lagrangianPotentials_node_gt=lagrangianPotentials_node*groundTruth';

groundTruth_features_pairwise=feature_pairwise_generator_binary(groundTruth,feature_nodes,0);

templagrangianPotentials_pairwise_gt=zeros(n_nodes,n_nodes);
for slice = 1 :n_nodes
    for sl=1:n_nodes
        templagrangianPotentials_pairwise_gt(slice,sl)=(reshape(groundTruth_features_pairwise(slice,sl,:),1,weight_size_pairwise)*theta_pairwise)'; 
    end
end
lagrangianPotentials_piarwise_gt=(sum(sum(templagrangianPotentials_pairwise_gt)))/n_pairs;


[p_maximizer,p_minimizer,game_value_maximizer,s_maximizer_nodes,s_minimizer_nodes]= DOMMulti_binary(groundTruth,theta_nodes,feature_nodes,lagrangianPotentials_node,theta_pairwise,feature_nodes,lagrangianPotentials_pairwise);

objective_value_maximizer=(lagrangianPotentials_node_gt)+(lagrangianPotentials_piarwise_gt)+game_value_maximizer(1);


maximizer_size=size(s_maximizer_nodes,1);
maximizer_expectation_pairwise=zeros(n_nodes,n_nodes,weight_size_pairwise);

for id=1:maximizer_size
    maximizer_expectation_pairwise=maximizer_expectation_pairwise+(p_maximizer(id)*feature_pairwise_generator_binary(double(s_maximizer_nodes(id,:)),feature_nodes,0));
end
sample_grad_node=((groundTruth)-(p_maximizer'*double(s_maximizer_nodes)))*feature_nodes';

sample_grad_pairwise=(groundTruth_features_pairwise)-maximizer_expectation_pairwise;


end



