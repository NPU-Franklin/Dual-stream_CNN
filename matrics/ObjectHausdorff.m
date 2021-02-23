function  objHausdorff = ObjectHausdorff(S,G)
% ObjectHausdorff calculates ojbect-level hausdorff distance between segmented objects
% and its corresponding ground truth objects
%
% Inputs:
%   S: a label image contains segmented objects
%   G: a label image contains ground truth objects
%
% Outputs:
%   objHausdorff: object-level Hausdorff distance
%
% Korsuk Sirinukunwattana
% BIAlab, Department of Computer Science, University of Warwick
% 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% convert S and G to the same format
S = single(S);
G = single(G);

% calculate total area and list of objects
tempS = S > 0;
totalAreaS = sum(tempS(:));

tempG = G > 0;
totalAreaG = sum(tempG(:));

listLabelS = unique(S);             % a list of labels of objects in S
listLabelS(listLabelS == 0) = [];

listLabelG = unique(G);             % a list of labels of objects in G
listLabelG(listLabelG == 0) = [];

% calculate object-level Hausdorff distance
temp1 = 0;                          % omega_i*H(G_i,S_i)

for iLabelS = 1:length(listLabelS)
    Si = S == listLabelS(iLabelS);
    intersectlist = G(Si);
    intersectlist(intersectlist == 0) = [];
    
    if ~isempty(intersectlist)
        indexGi = mode(intersectlist);
        Gi = G == indexGi;
    else
        tempDist = zeros(length(listLabelG),1);
        for iLabelG = 1:length(listLabelG)
            Gi = G == listLabelG(iLabelG);
            tempDist(iLabelG) = Hausdorff(Gi,Si);
        end
        [~,minIdx] = min(tempDist);
        Gi = G == listLabelG(minIdx);
    end
    
    omegai = sum(Si(:))/totalAreaS;
    temp1 = temp1 + omegai*Hausdorff(Gi,Si);
end


temp2 = 0;                          % tilde_omega_i*H(tilde_G_i,tilde_S_i)

for iLabelG = 1:length(listLabelG)
    tildeGi = G == listLabelG(iLabelG);
    intersectlist = S(tildeGi);
    intersectlist(intersectlist == 0) = [];

    if ~isempty(intersectlist)
        indextildeSi = mode(intersectlist);
        tildeSi = S == indextildeSi;
    else
        tempDist = zeros(length(listLabelS),1);
        for iLabelS = 1:length(listLabelS)
            tildeSi = S == listLabelS(iLabelS);
            tempDist(iLabelS) = Hausdorff(tildeGi,tildeSi);
        end
        [~,minIdx] = min(tempDist);
        tildeSi = S == listLabelS(minIdx);
    end
    
    tildeOmegai = sum(tildeGi(:))/totalAreaG;
    temp2 = temp2 + tildeOmegai*Hausdorff(tildeGi,tildeSi);
end


objHausdorff = (temp1 + temp2)/2;

end

function hausdorffDistance = Hausdorff(S,G)
% Hausdorff calculates hausdorff distance between segmented objects in S
% and ground truth objects in G
%
% Inputs:
%   S: a label image contains segmented objects
%   G: a label image contains ground truth objects
%
% Outputs:
%   hausdorffDistance: as the name indicated
%
% Korsuk Sirinukunwattana
% BIAlab, Department of Computer Science, University of Warwick
% 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% convert S and G to the same format
S = single(S);
G = single(G);

% check if S or G is non-empty

listS = unique(S);            % list of labels of segmented objects
listS(listS == 0) = [];       % remove the label of the background
numS = length(listS);         % the total number of segmented objects in S

listG = unique(G);            % list of labels of ground truth objects
listG(listG == 0) = [];       % remove the label of the background
numG = length(listG);         % the total number of ground truth in G

if numS == 0 && numG == 0    % no segmented object & no ground truth objects
    hausdorffDistance = 0;
    return
elseif numS == 0 || numG == 0
    hausdorffDistance = Inf;
    return
else
    % do nothing
end

% Calculate Hausdorff distance
maskS = S > 0;
maskG = G > 0;
[rowInd,colInd] = ind2sub(size(S),1:numel(S));
coordinates = [rowInd',colInd'];

x = coordinates(maskG,:);
y = coordinates(maskS,:);

% sup_{x \in G} inf_{y \in S} \|x-y\|
[~,dist] = knnsearch(y,x);
dist1 = max(dist);

% sup_{x \in S} inf_{y \in G} \|x-y\|
[~,dist] = knnsearch(x,y);
dist2 = max(dist);

hausdorffDistance = max(dist1,dist2);

end