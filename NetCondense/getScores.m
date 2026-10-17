function getScores(graphFile)

[temporalGraph, n, t ]= getTemporalGraph(graphFile);
unionGraph = getUnionGraph(temporalGraph, n, t);
flattenedGraph = flattenNetwork(temporalGraph, n, t);
flatM = (flattenedGraph+ flattenedGraph')/2;
[u, lambda] = eigs(flatM,1,'LM');
[v, lambda] = eigs(flatM',1, 'LM');

u = abs(u);
v = abs(v);

activeTimes = 1:t;

Flat2 = sparse(n,n*t);
for i = 1:t
    Flat2(:,((i-1)*n)+1:(i*n)) = temporalGraph{i};
end



lambda = abs(lambda)

%edgescores = getNodeScores(flatM, unionGraph, n, t, u, v, lambda);
%edgescores = tempNodesScore(flatM, unionGraph, n, t, u, v, lambda);
tic;

edgescores = getNodeScores(Flat2, unionGraph, n, t, u, v, lambda);
timescores = getTimeScores(flatM, activeTimes,n, t, u, v, lambda);
timeToScore = toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Print scores
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
timefile = fopen(strcat(graphFile,'_ScoreTimeFlat2'),'w');
fprintf(timefile, '%.20f',timeToScore);
fclose(timefile);

outfile = fopen(strcat(graphFile,'_edgeScores'),'w');  
for scoreIndex = 1: size(edgescores,1)
     fprintf(outfile,'%d\t%d\t%.4f\n', [edgescores(scoreIndex,1),edgescores(scoreIndex,2),edgescores(scoreIndex,3)]);
end
fclose(outfile);




tempoutfile = fopen(strcat(graphFile,'_timeScores'),'w');
for scoreIndex = 1: size(timescores,1)
    fprintf(tempoutfile,'%d\t%d\t%.4f\n', [timescores(scoreIndex,1),timescores(scoreIndex,2), timescores(scoreIndex,3)]);
end
fclose(tempoutfile);

%exit;
end
