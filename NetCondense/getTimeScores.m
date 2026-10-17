function[timescores] = getTimeScores(flattenedGraph, activeTimes,n, t, u,v, lambda)

uv = u'*v;

timescores = zeros(t-1, 3);
for i = 1: length(activeTimes)-1
    index = activeTimes(i);
    index1 = activeTimes(i+1);
%     if(i == length(activeTimes))
%         index1 = activeTimes(1);
%     else
%         index1 = activeTimes(i+1);
%     end
    TNodes = linspace( ((index-1)*n) +1, (n*(index)), n);
    T1Nodes = linspace( ((index1-1)*n) +1, (n*(index1)), n);

    ui = u(TNodes);
    vi = v(TNodes);
    uj = u(T1Nodes);
    vj = v(T1Nodes);
    firstterm = ui'*vi + uj'*vj;
    smallF = flattenedGraph(TNodes, T1Nodes);
    smallF = (smallF+smallF');
    secondTerm = ui' * smallF * vj;
    thirdTerm = (sum((lambda * ui'*vi)+ (lambda * uj'*vj)))/2;
    score = ((-lambda * firstterm) + secondTerm + thirdTerm)/ (uv - firstterm);
     timescores(i,:) = [index,index1, abs(score)];
     i
     t
    
end
timescores = sortrows(timescores,3);
timescores(all(timescores==0,2),:)=[];

