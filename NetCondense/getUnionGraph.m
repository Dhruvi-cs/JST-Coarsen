function [unionGraph] = getUnionGraph(temporalGraph, n, t)
%function [unionGraph] = getUnionGraph(graphFile)

%[temporalGraph, n, t ]= getTemporalGraph2(graphFile);

unionGraph = sparse(n,n);
for i =1:t
    unionGraph = unionGraph + temporalGraph{i};
end

unionGraph = spones(unionGraph);