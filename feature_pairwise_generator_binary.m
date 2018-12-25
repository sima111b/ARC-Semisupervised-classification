


function feature_pairwise=feature_pairwise_generator_binary(nodes,features,general)
%gt=1 the general pairwise features assuming or labeld 1
%we assume feature_pairwise is additive and is non-zero only between the
%samples which have different nodes.
global n_nodes;
global weight_size_pairwise;
cValue=1;
n_features=size(features,2); 
feature_pairwise=zeros(n_nodes,n_nodes,weight_size_pairwise);
for i=1:n_nodes
    for j=1:n_nodes
        if (i==j)
            feature_pairwise(i,j,:)=0;
        else
                if ((nodes(i)~=nodes(j))|| (general==1))
                    feature_pairwise(i,j,:)=1./(abs(features(:,i)- features(:,j))+cValue);
                end
        end
    end
end


end



