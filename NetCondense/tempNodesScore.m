%This function is called by a function wrapper, that calls this fucntion
%and prints the score
%input : 
%flattenedGraph: sparse flattend graph : (F+F')/2
%unionGraph : union Graph of the temporal network
% n,t :  no of nodes, timestamps
% u, v, lambda : left and right eigenvector of flattenedGraph and its
% eigenalue respectively

function[nodescores] = tempNodesScore(flattenedGraph, unionGraph, n, t, u,v, lambda)

[i,j,val] = find(unionGraph);

uv = u'*v;
nodescores = zeros(length(i), 3);

uvProd = sum(uv);
%parpool(4)

for index = 1: length(i)
    iNodes = linspace( i(index), (n*(t-1)+i(index)), t);
    jNodes = linspace(j(index), (n*(t-1)+j(index)), t);
    ui = u(iNodes);
    vi = v(iNodes);
    uj = u(jNodes);
    vj = v(jNodes);
    
    firstterm = ui'*vi + uj'*vj;
    %% Alternate way of calculating secondTerm. But this is slower than the one below for large graphs
    %    smallF = flattenedGraph(iNodes, jNodes);
    %    secondTerm = ui' * smallF * vj;
    %%
    nonzers = (flattenedGraph(sub2ind(size(flattenedGraph),iNodes,circshift(jNodes,1,1))))';
    nonzers1 = (flattenedGraph(sub2ind(size(flattenedGraph),jNodes,circshift(iNodes,1,1))))';
    secondTerm = sum((ui .* nonzers ).* vj);
    %%
    % sub2ind does not work for DBLP, because the no of indices in (F+F')
    % is greater than max value of int32.
    % Do you have any idea how to deal with it ?
    % Looping makes it worse than the slower method in previous comment.
    %%
    
    thirdTermVec = (lambda * ui) - (nonzers .*circshift(uj,1,1)) + (lambda * uj) - (nonzers1 .*circshift(ui,1,1));
    %thirdTermVec = (lambda * ui) - (smallF * uj([t,1:t-1],:)) + (lambda * uj) - (smallF'*ui([t,1:t-1],:));
    %thirdTermVec = (lambda * ui) - (smallF * uj) + (lambda * uj) - (smallF'*ui);
    thirdTerm = sum(thirdTermVec'*uj)/2;
    score = ((-lambda * firstterm) + secondTerm + thirdTerm)/ (uvProd - firstterm);
    nodescores(index,:) = [i(index),j(index), abs(score)];

end
%p = gcp;
%delete(p)
nodescores = sortrows(nodescores,3);

