function [fG] = flattenNetwork(temporalGraph, n, t)


fG = sparse(n*t, n*t);



for timeStamp = 1:t
        
        [i,j,val] = find(temporalGraph{timeStamp}+speye(n));
        len = length(i);
        if timeStamp == t
            next = 1;
        else
            next = timeStamp+1;
        end
          
        val(len+1) = 0;
        i = i+(n*(timeStamp-1));
        i(len+1) = n*t;
        
        
        j = j+ (n*(next - 1));
        j(len+1) = n*t;
        temp = sparse(i,j, val);
        
        fG = fG + temp;
end
%imagesc(fG)
%spy(fG)
%drawnow
end