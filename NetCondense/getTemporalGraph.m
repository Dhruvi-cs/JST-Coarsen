function [temporalGraph, n, t] = getTemporalGraph(graphFile)

allEdgelists = load(graphFile);
n = max(max(allEdgelists(:,2)), max(allEdgelists(:,3)));
t = max(allEdgelists(:,1));
temporalGraph = cell(t,1);
for i=1:t
    tempmatrix = allEdgelists(allEdgelists(:,1)==i, 2:5);
    tempmatrix(end+1,:)=[n n 0 0];
    firstmatrix = spconvert(tempmatrix(:,1:3));
    secondmatrix = spconvert(tempmatrix(:,[2,1,4]));
    
    temporalGraph{i} = firstmatrix +  secondmatrix;
end

return
    