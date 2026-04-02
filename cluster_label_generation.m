%% ================================================================
%  解耦法：先在样本空间选 K（与 SOM 无关），再训练 SOM 初始化 K-means 输出
%  - 读取 Excel 的“数据区”（行=指标 D，列=样本 Q）
%  - 纯 K-means++ 多重复：Silhouette / CH / DBI 评估择 K（解耦于 SOM）
%  - SOM 多次重启（含 seed=200，与扫参一致），自动选择 TE 最小的网络
%  - 六边形按“命中样本簇众数”着色；标签显示样本名
%  - 导出 som_kmeans_labels.csv 与 som_cluster_groups.xlsx
%  依赖：
%    Neural Network Toolbox: selforgmap / plotsom*
%    Statistics and ML Toolbox: kmeans / evalclusters / pca / pdist / squareform
% ================================================================
clear; clc; close all;

%% -------------------- 参数区 --------------------
excelFile       = 'test-西山岩溶域数据.xlsx';   % ← 按需修改
colRangeData    = 'B2:HW21';                   % ← 按需修改（D×Q：行=指标，列=样本）

% 1) 选 K（与 SOM 解耦）的设置
Klist           = 2:50;                        % 候选 K（将由 [PATCH B] 动态裁剪）
seedSelect      = 777;                         % 选 K 阶段随机源
replicatesSel   = 30;                          % 选 K 阶段每次 kmeans 重复次数
primary_metric  = 'silhouette';                % 'silhouette' | 'ch' | 'dbi'

% 2) SOM 形状与训练设置
som_dim1        = 7;                           % SOM 网格尺寸（行）
som_dim2        = 11;                          % SOM 网格尺寸（列）
auto_pick_shape = false;                       % true 则使用 PCA-aspect 推荐网格
% —— 扫参与最终训练使用**同一基准种子**：包含 seed=200（与你扫参一致）——
seedSOM_scan_base  = 200;                      % 扫参基准种子
seedSOM_final_base = 200;                      % 最终多重启动的基准种子（含 200）
som_repeats        = 20;                       % 多重启动次数（种子=base+(0:repeats-1)）
coverSteps         = 200;                      % 建议与扫参一致，保证口径一致
initNeighbor       = 5;
epochsFine         = 200;

% 3) 最终 K-means 设置
seedFinal       = 1;                           % 最终 K-means 随机源（与 K 相加）

%% -------------------- 以“数据区”为准读取 --------------------
Xfull   = readmatrix(excelFile, 'Range', colRangeData);   % 只读数值区
numRows = size(Xfull, 1);
numCols = size(Xfull, 2);

lastHdrColLetter = excel_col(1 + numCols);               % B 起宽度=numCols -> 末列表头
hdrCells     = readcell(excelFile, 'Range', sprintf('B1:%s1', lastHdrColLetter));
sample_names = string(hdrCells(:));                       % 长度 = numCols

rowNameCells    = readcell(excelFile, 'Range', sprintf('A2:A%d', 1 + numRows));
indicator_names = string(rowNameCells(:));                % 长度 = numRows

% 空名补齐（不删列、不删行）
ms = (ismissing(sample_names)   | sample_names   == ""); si = find(ms);
if ~isempty(si), sample_names(si)   = compose("S%03d", si); end
mi = (ismissing(indicator_names) | indicator_names == ""); ii = find(mi);
if ~isempty(ii), indicator_names(ii) = compose("Idx%03d", ii); end

X = Xfull;  [D, Q] = size(X);

%% -------------------- 指标诊断 --------------------
if ~isnumeric(X), X = str2double(string(X)); end
allNaN_row  = all(isnan(X), 2);
zeroVar_row = false(D,1);
for i = 1:D
    xi = X(i, :); xi = xi(~isnan(xi));
    zeroVar_row(i) = isempty(xi) || (std(xi) < 1e-12);
end

fprintf('\n—— 指标诊断 ——\n');
fprintf('总指标行: %d  | 全NaN: %d  | 零方差(未剔除): %d\n', D, sum(allNaN_row), sum(zeroVar_row));
writetable(table(indicator_names, allNaN_row, zeroVar_row, ...
    'VariableNames', {'indicator','allNaN','zeroVariance'}), ...
    'indicator_diagnosis.xlsx');
disp('已导出：indicator_diagnosis.xlsx');

% 仅剔除全NaN行（零方差行保留，但标准化时保护）
valid_rows      = ~allNaN_row;
X               = X(valid_rows, :);
indicator_names = indicator_names(valid_rows);
[D, Q]          = size(X);

%% -------------------- 稳健标准化（忽略 NaN；std=0 保护） --------------------
x  = zscore_ignore_nan(X);    % D×Q（标准化后用于 SOM & 聚类）
Xq = x.';                     % Q×D（样本×特征）
fprintf('读取完成并标准化：%d 个指标 × %d 个样本（D×Q）\n', D, Q);

% ===== [PATCH B] 收紧 K 的最大上限，弱化 DBI 对大K的偏好 =====
Klist_orig = Klist;                           % 记录原始候选，以便打印对比
Kmax = min([15, floor(sqrt(Q)), max(3, floor(0.10*Q))]);  % Q=样本数 动态取区间
%Kmax = 15;
Klist = Klist(Klist >= 2 & Klist <= Kmax);
fprintf('K 候选范围：原始 [%d, %d] → 裁剪为 [%d, %d] (Q=%d, Kmax=%d)\n', ...
    min(Klist_orig), max(Klist_orig), Klist(1), Klist(end), Q, Kmax);

%% -------------------- PCA 长宽比建议（可选启用自动网格） --------------------
[~,~,latent] = pca(Xq); 
aspect   = sqrt(latent(1)/latent(2));         % 标准差比
M_target = round(5*sqrt(size(Xq,1)));         % Vesanto 目标神经元数
shapes = [8 9; 8 10; 9 9; 7 11; 6 13; 10 7; 11 7; 12 7];
scores = zeros(size(shapes,1),1);
for i=1:size(shapes,1)
    d1 = shapes(i,1); d2=shapes(i,2);
    M  = d1*d2;      % 网格神经元数
    ar = max(d1,d2)/min(d1,d2);               % 网格长宽比
    scores(i) = 0.6*abs(M - M_target) + 0.4*abs(ar - aspect)*M_target;
end
[~,best] = min(scores);
fprintf('PCA-aspect 推荐网格：%dx%d（目标M≈%d, aspect≈%.2f）\n', ...
    shapes(best,1), shapes(best,2), M_target, aspect);
if auto_pick_shape
    som_dim1 = shapes(best,1); som_dim2 = shapes(best,2);
    fprintf('已启用自动形状，SOM 网格改为：%dx%d\n', som_dim1, som_dim2);
end

%% -------------------- 解耦：在样本空间选 K（与 SOM 无关） --------------------
clusterer_plain = @(X_,K_) kmeans_seeded_plain(X_, K_, seedSelect, replicatesSel);

eva_sil = evalclusters(Xq, clusterer_plain, 'silhouette',       'KList', Klist);
eva_ch  = evalclusters(Xq, clusterer_plain, 'CalinskiHarabasz', 'KList', Klist);
eva_dbi = evalclusters(Xq, clusterer_plain, 'DaviesBouldin',    'KList', Klist);

% ===== [PATCH C] DBI*（带惩罚的 DBI）以弱化大K优势 =====
insK     = eva_dbi.InspectedK(:);
dbi_vals = eva_dbi.CriterionValues(:);
alpha    = 0.02;   % 惩罚强度：可在 0.01 ~ 0.05 区间微调
dbi_star = dbi_vals .* (1 + alpha*(insK - 1));
[~, ix_star] = min(dbi_star);
bestK_dbi_star = insK(ix_star);

fprintf('【解耦选K】Silhouette最佳K=%d, CH最佳K=%d, DBI最佳K=%d\n', eva_sil.OptimalK, eva_ch.OptimalK, eva_dbi.OptimalK);
fprintf('【DBI*】alpha=%.2f | 最优K=%d | DBI*最小=%.6f | 原DBI最小=%.6f\n', ...
    alpha, bestK_dbi_star, dbi_star(ix_star), min(dbi_vals));

bestK_sil = eva_sil.OptimalK;
bestK_ch  = eva_ch.OptimalK;
bestK_dbi = eva_dbi.OptimalK;

% —— 仲裁：仍以 primary_metric 为主；若 sil 与 ch 一致，则以其一致覆盖 —— 
switch lower(primary_metric)
    case 'silhouette'
        K0 = bestK_sil;
        if K0 ~= bestK_ch && bestK_ch == bestK_dbi, K0 = bestK_ch; end
    case 'ch'
        K0 = bestK_ch;
        if K0 ~= bestK_sil && bestK_sil == bestK_dbi, K0 = bestK_sil; end
    case 'dbi'
        % ===== [PATCH C] 当选择 DBI 为主判据时，使用 DBI* 的最优K =====
        K0 = bestK_dbi_star;
        if K0 ~= bestK_sil && bestK_sil == bestK_ch, K0 = bestK_sil; end
    otherwise
        K0 = bestK_sil;
end
fprintf('按“%s”为主判据，最终选定 K0 = %d（与 SOM 解耦）\n', upper(primary_metric), K0);

%% ========== 可视化：Silhouette / CH / DBI / DBI* 随 K 的曲线 ==========
% 需放在 eva_sil / eva_ch / eva_dbi、bestK_*、K0 都已就绪之后

% 1) 抽取曲线数据
K_grid   = eva_sil.InspectedK(:);                 % 与 Klist 一致
sil_vals = eva_sil.CriterionValues(:);            % 越大越好
ch_vals  = eva_ch.CriterionValues(:);             % 越大越好
dbi_vals = eva_dbi.CriterionValues(:);            % 越小越好

% 与主代码保持一致的 DBI* 计算（如果上面你改了 alpha，这里也同步）
alpha    = 0.02;
dbi_star = dbi_vals .* (1 + alpha*(K_grid - 1));

% 2) 指标最优 K
K_sil  = eva_sil.OptimalK;
K_ch   = eva_ch.OptimalK;
K_dbi  = eva_dbi.OptimalK;                        % 原始 DBI
% 你在上文已算过 bestK_dbi_star，这里再保险一次：
[~, ix_star] = min(dbi_star);
K_dbi_star   = K_grid(ix_star);

% 3) 三联图（各指标各自 y 轴）
fig1 = figure('Color','w','Name','K 选择三联图','Position',[120 120 1100 750]);

% --- (a) Silhouette ---
subplot(3,1,1);
plot(K_grid, sil_vals,'-o','LineWidth',1.5,'MarkerSize',5); hold on; grid on;
xline(K_sil,'-','Color',[0.2 0.6 0.2],'LineWidth',1.2,'Label','Sil 最优 K','LabelOrientation','horizontal');
xline(K0,'--','Color',[0.4 0.4 0.4],'LineWidth',1.2,'Label','最终采用 K0','LabelOrientation','horizontal');
ylabel('Silhouette');
title('Silhouette vs. K（越大越好）');

% --- (b) Calinski–Harabasz ---
subplot(3,1,2);
plot(K_grid, ch_vals,'-o','LineWidth',1.5,'MarkerSize',5); hold on; grid on;
xline(K_ch,'-','Color',[0.2 0.6 0.8],'LineWidth',1.2,'Label','CH 最优 K','LabelOrientation','horizontal');
xline(K0,'--','Color',[0.4 0.4 0.4],'LineWidth',1.2,'Label','最终采用 K0','LabelOrientation','horizontal');
ylabel('CH 指标');
title('Calinski–Harabasz vs. K（越大越好）');

% --- (c) DBI & DBI* ---
subplot(3,1,3);
plot(K_grid, dbi_vals,'-o','LineWidth',1.5,'MarkerSize',5); hold on; grid on;
plot(K_grid, dbi_star,'-s','LineWidth',1.5,'MarkerSize',5);
xline(K_dbi,'-','Color',[0.85 0.33 0.10],'LineWidth',1.2,'Label','DBI 最优 K','LabelOrientation','horizontal');
xline(K_dbi_star,':','Color',[0.49 0.18 0.56],'LineWidth',1.2,'Label','DBI* 最优 K','LabelOrientation','horizontal');
xline(K0,'--','Color',[0.4 0.4 0.4],'LineWidth',1.2,'Label','最终采用 K0','LabelOrientation','horizontal');
ylabel('DBI / DBI*');
xlabel('聚类数 K');
title(sprintf('Davies–Bouldin vs. K（越小越好；DBI* α=%.2f）', alpha));
legend({'DBI','DBI*'},'Location','best');

% 保存三联图
try
    exportgraphics(fig1,'k_selection_curves_triple.png','Resolution',300);
catch
    saveas(fig1,'k_selection_curves_triple.png');
end

% 4) 叠加图（把四条曲线标准化到 [0,1]，方向已统一为“越大越好”）
% 为了同轴可比：对 CH、Sil 做 [0,1] 归一；对 DBI/DBI* 先取反向(越小越好→越大越好)，再归一
norm01 = @(v) (v - min(v)) ./ max(eps, (max(v) - min(v)));
sil_n  = norm01(sil_vals);
ch_n   = norm01(ch_vals);
dbi_n  = norm01(-dbi_vals);          % 取负号后越大越好
dbiS_n = norm01(-dbi_star);

fig2 = figure('Color','w','Name','K 选择叠加图（归一化）','Position',[120 120 900 500]);
plot(K_grid, sil_n,'-o','LineWidth',1.5,'MarkerSize',5); hold on; grid on;
plot(K_grid, ch_n,'-o','LineWidth',1.5,'MarkerSize',5);
plot(K_grid, dbi_n,'-o','LineWidth',1.5,'MarkerSize',5);
plot(K_grid, dbiS_n,'-o','LineWidth',1.5,'MarkerSize',5);

% 标注各自最优 K（在归一化图上以竖线标示）
xline(K_sil,'-','Color',[0.2 0.6 0.2],'LineWidth',1.0,'Label','Sil 最优');
xline(K_ch,'-','Color',[0.2 0.6 0.8],'LineWidth',1.0,'Label','CH 最优');
xline(K_dbi,'-','Color',[0.85 0.33 0.10],'LineWidth',1.0,'Label','DBI 最优');
xline(K_dbi_star,':','Color',[0.49 0.18 0.56],'LineWidth',1.0,'Label','DBI* 最优');
xline(K0,'--','Color',[0.4 0.4 0.4],'LineWidth',1.2,'Label','最终 K0');

xlabel('聚类数 K');
ylabel('归一化指标值（越大越好）');
title('Sil / CH / DBI / DBI* 归一化叠加对比');
legend({'Silhouette','CH','-DBI (norm)','-DBI* (norm)'},'Location','best');

% 保存叠加图
try
    exportgraphics(fig2,'k_selection_curves_overlay.png','Resolution',300);
catch
    saveas(fig2,'k_selection_curves_overlay.png');
end



%% —— 可选人工覆盖 K0 ——（放在“最终选定 K0 = …”之后）
kmin = min(Klist); kmax = max(Klist);
prompt = sprintf(['检测到最佳 K = %d（基于 %s）。\n' ...
                  '如需人工覆盖，请输入整数 K ∈ [%d, %d]。\n' ...
                  '留空或取消将使用最佳 K。'], ...
                  K0, upper(primary_metric), kmin, kmax);

answer = inputdlg(prompt, '确认聚类 K', 1, {num2str(K0)});
if isempty(answer) || isempty(strtrim(answer{1}))
    K_user = K0;   % 取消或留空 → 用最佳K
else
    K_user = str2double(answer{1});
    if ~isfinite(K_user) || (round(K_user) ~= K_user)
        warndlg('输入不是有效整数，已使用最佳 K。', '提示');
        K_user = K0;
    end
end

% 范围裁剪
if K_user < kmin || K_user > kmax
    warndlg(sprintf('输入 K 超出范围 [%d,%d]，已裁剪到边界。', kmin, kmax), '提示');
    K_user = max(kmin, min(kmax, round(K_user)));
end

K0 = round(K_user);
fprintf('>>> 最终采用的 K = %d（可人工覆盖）\n', K0);




%% ============== SOM 网格扫参：比较 QE / TE 选形状（可选） ==============
candidates = [8 9; 8 10; 9 9; 7 11; 10 8];
seedSOM_scan      = seedSOM_scan_base;        % 与最终基准保持一致（200）
coverSteps_scan   = 200;                      % 与最终设置一致
initNeighbor_scan = 5;
epochsFine_scan   = 200;

results = table('Size',[size(candidates,1) 4], ...
    'VariableTypes', {'double','double','double','double'}, ...
    'VariableNames', {'dim1','dim2','QE','TE'});

for ii = 1:size(candidates,1)
    d1 = candidates(ii,1); d2 = candidates(ii,2);
    rng(seedSOM_scan,'twister');
    net_scan = selforgmap([d1 d2], coverSteps_scan, initNeighbor_scan, 'hextop', 'linkdist');
    net_scan.inputs{1}.processFcns = {};      % 禁用 mapminmax
    net_scan.trainParam.epochs = epochsFine_scan;
    [net_scan,~] = train(net_scan, x);

    Wscan  = net_scan.IW{1};                  % M×D
    yscan  = net_scan(x);
    bmu    = vec2ind(yscan);                  % 1×Q
    pos    = net_scan.layers{1}.positions.';  % M×2

    % QE
    QE = mean( sqrt( sum( (Xq - Wscan(bmu,:)).^2 , 2) ), 'omitnan' );

    % TE（自适应邻接阈值）
    P = squareform(pdist(pos));
    nz = P(P>0);
    base = prctile(nz, 1);                    % 最近邻典型尺度（hextop≈1）
    adj  = (P > 0) & (P <= 1.10*base);

    Dwx = pdist2(Wscan, Xq, 'euclidean');     % M×Q
    [~, order] = sort(Dwx, 1, 'ascend');
    b1 = order(1, :).'; 
    b2 = order(2, :).';
    notNeighbor = arrayfun(@(i,j) ~adj(i,j), b1, b2);
    TE = mean(notNeighbor);

    results{ii, :} = [d1, d2, QE, TE];
end
disp('SOM 网格扫参（QE越小越好，TE越小越好）:');
disp(results);

%% -------------------- 多次重启训练 SOM（含 seed=200），自动挑选 TE 最优 --------------------
seedSOM_base = seedSOM_final_base;            % 200
[net, W, bmuIdx, qe_best, te_best, best_seed, te_fixed_best, base_cut, cut_off] = ...
    train_som_multistart(Xq, x, som_dim1, som_dim2, seedSOM_base, som_repeats, ...
                         coverSteps, initNeighbor, epochsFine);

fprintf('输入预处理 processFcns: '); disp(net.inputs{1}.processFcns);   % 应为空 {}
fprintf('最终采用的 SOM 种子 = %d（在 %d 次重启中 TE 最小）\n', best_seed, som_repeats);
fprintf('SOM 量化误差 QE = %.6f\n', qe_best);
fprintf('SOM 拓扑误差 TE = %.6f（自适应口径）\n', te_best);
fprintf('[调试] TE_fixed=%.6f, base=%.4f, cutoff=%.4f\n', te_fixed_best, base_cut, cut_off);

% 可视化：SOM 命中图
figure('Color','w'); plotsomhits(net, x);
title(sprintf('SOM 命中图（hits），%d×%d 网格（seed=%d）', som_dim1, som_dim2, best_seed));

%% -------------------- 最终聚类：在 W 上取初始中心 → 在 Xq 上精化（用 K0） --------------------
rng(seedFinal + K0, 'twister');
[~, Cw] = kmeans(W, K0, 'Distance','sqeuclidean', 'Replicates', 20, 'Start','plus', 'EmptyAction','singleton');
[idxX, Cx] = kmeans(Xq, K0, 'Distance','sqeuclidean', 'Start', Cw, 'MaxIter', 300, 'EmptyAction','singleton');
clusterLabel = idxX(:);

% 报告最终 DBI（针对“最终输出”的客观衡量）
final_dbi = dbi_euclidean(Xq, clusterLabel, Cx);
fprintf('最终输出（SOM初始化+样本精化，K=%d）的 DBI = %.6f\n', K0, final_dbi);

%% ===== 可选：诊断打印（你之前用过，保留便于排查） =====
K = K0;
counts = accumarray(clusterLabel(:),1,[K 1],@sum,0);
fprintf('[DBG] 簇规模最小=%d 中位=%g 最大=%d | 空簇=%d\n', ...
    min(counts), median(counts), max(counts), sum(counts==0));

s = zeros(K,1);
for i=1:K
    Xi = Xq(clusterLabel==i,:);
    if isempty(Xi), s(i)=NaN; else
        s(i) = mean( sqrt(sum((Xi - Cx(i,:)).^2,2)) );
    end
end
Cdist = squareform(pdist(Cx,'euclidean')); Cdist(eye(K)==1) = inf;
Rij = (s + s.') ./ Cdist; Rij(eye(K)==1) = -inf;
[Ri, jstar] = max(Rij, [], 2); [dbi_max, istar] = max(Ri);
fprintf('[DBG] max Rij=%.6f 出现在 (i=%d, j=%d)\n', dbi_max, istar, jstar(istar));
d_c = norm(Cx(istar,:) - Cx(jstar(istar),:));
fprintf('[DBG] 该对簇: s_i=%.6f, s_j=%.6f, |c_i-c_j|=% .6e\n', s(istar), s(jstar(istar)), d_c);
feat_std = std(Xq,0,1); 
fprintf('[DBG] 标准化后特征std: min=%.3f, median=%.3f, max=%.3f\n', min(feat_std), median(feat_std), max(feat_std));

%% -------------------- 六边形无缝着色（上色=命中样本簇标签众数） --------------------
% pos  = net.layers{1}.positions.';     % M×2
% Dmat = squareform(pdist(pos)); Dmat(Dmat==0) = inf;
% dnn  = median(min(Dmat,[],2));
% theta   = (0:6)'*pi/3 + pi/6;         % 平顶
% hexUnit = [cos(theta) sin(theta)];
% r       = dnn / sqrt(3);              % 边贴边
% 
%% 众数标签
% M = size(W,1);
% hitIds  = accumarray(bmuIdx(:), (1:size(Xq,1)).', [M 1], @(v){v}, {[]});
% neuronLabel = zeros(M,1); % 0 = 无命中
% for i = 1:M
%     ids = hitIds{i};
%     if ~isempty(ids)
%         neuronLabel(i) = mode(clusterLabel(ids));
%     end
% end
% 
% cmap       = lines(K0);
% emptyColor = [0.85 0.85 0.85];
% figure('Color','w'); hold on;
% 
% for i = 1:M
%     verts = pos(i,:) + r*hexUnit;
%     faceColor = emptyColor;
%     if neuronLabel(i) >= 1
%         faceColor = cmap(neuronLabel(i),:);
%     end
%     patch(verts(:,1), verts(:,2), faceColor, ...
%         'EdgeColor',[0.35 0.35 0.45], 'LineWidth',0.8, 'FaceAlpha',0.95);
% 
%%     标签用“样本名”
%     ids = hitIds{i};
%     if ~isempty(ids)
%         Nshow = 4;   % 每格最多显示 N 个样本名
%         names_i = string(sample_names(ids)).';
%         if numel(names_i) > Nshow
%             txt = strjoin(names_i(1:Nshow), ',') + "…";
%         else
%             txt = strjoin(names_i, ',');
%         end
%         text(pos(i,1), pos(i,2), txt, ...
%             'HorizontalAlignment','center','VerticalAlignment','middle', ...
%             'FontSize',6, 'Color','w', 'FontWeight','bold');
%     end
% end
% axis equal off
% title(sprintf('SOM 六边形（SOM初始化的样本级K-means，上色=众数，K=%d，解耦选K）', K0));
% 
%% 图例
% hleg = gobjects(K0 + 1, 1);
% for k = 1:K0
%     hleg(k) = plot(nan, nan, 's', 'MarkerFaceColor', cmap(k,:), ...
%                    'MarkerEdgeColor','none', 'MarkerSize', 8);
% end
% hleg(K0+1) = plot(nan, nan, 's', 'MarkerFaceColor', emptyColor, ...
%                  'MarkerEdgeColor','none', 'MarkerSize', 8);
% legend(hleg, [arrayfun(@(k) sprintf('Cluster %d',k), 1:K0, 'uni',0), {'No hits'}], ...
%        'Location','eastoutside');




%% -------------------- 导出（样本标签 + 分组Excel） --------------------
% 样本名与 Q 对齐
if numel(sample_names) ~= size(Xq,1)
    if numel(sample_names) > size(Xq,1)
        sample_names = sample_names(1:size(Xq,1));
    else
        sample_names(end+1:size(Xq,1),1) = compose("S%03d", numel(sample_names)+1:size(Xq,1));
    end
end

% 1) 每样本标签
T_labels = table(sample_names(:), clusterLabel(:), ...
    'VariableNames', {'sample_id','cluster'});
writetable(T_labels, 'som_kmeans_labels.csv');
disp('已导出：som_kmeans_labels.csv');

% 2) 分组 Excel
counts    = accumarray(clusterLabel(:), 1, [K0 1], @sum, 0);
T_summary = table((1:K0).', counts, 'VariableNames', {'cluster','count'});
writetable(T_summary, 'som_cluster_groups.xlsx', 'Sheet', 'Summary', 'WriteMode', 'overwritesheet');

T_groups = sortrows(T_labels, 'cluster');
writetable(T_groups, 'som_cluster_groups.xlsx', 'Sheet', 'Groups', 'WriteMode', 'overwritesheet');

for k = 1:K0
    sel = (clusterLabel == k);
    T_k = table(sample_names(sel), repmat(k,sum(sel),1), ...
                'VariableNames', {'sample_id','cluster'});
    writetable(T_k, 'som_cluster_groups.xlsx', 'Sheet', sprintf('Cluster_%d', k), 'WriteMode', 'overwritesheet');
end
disp('已导出：som_cluster_groups.xlsx（Summary、Groups、各簇工作表）');

% view(net);  % 如需查看网络结构


%% ===================== 本地函数 =====================
function Xz = zscore_ignore_nan(X)
% 对 X (D×Q) 的每一行做 z-score（忽略 NaN；std=0 保护；NaN→0）
    [D, Q] = size(X);
    Xz = zeros(D, Q);
    for i = 1:D
        xi = X(i, :);
        mu = mean(xi, 'omitnan');
        sd = std(xi, 0, 'omitnan');
        if ~isfinite(sd) || sd < 1e-12, sd = 1; end
        zi = (xi - mu) ./ sd;
        zi(~isfinite(zi)) = 0;   % NaN/Inf → 0（均值）
        Xz(i, :) = zi;
    end
end

function idx = kmeans_seeded_plain(X, K, seed, replicatesSel)
% 选 K 阶段用的纯 K-means++ 聚类器（与 SOM 解耦）
% - 每个 K 固定随机源（seed+K），K-means++ 起点，多重复
    rng(seed + K, 'twister');
    idx = kmeans(X, K, ...
        'Distance','sqeuclidean', ...
        'Start','plus', ...
        'Replicates', replicatesSel, ...
        'MaxIter', 300, ...
        'EmptyAction','singleton');
end

function dbi = dbi_euclidean(Xq, labels, Cx)
% Davies–Bouldin Index（欧氏距离）——用于报告“最终输出”的客观指标
% Xq: Q×D（样本×特征）
% labels: Q×1 的簇标签，取值 1..K
% Cx: K×D 的簇中心
    K = size(Cx,1);
    s = zeros(K,1);                       % 簇内平均散度
    for i = 1:K
        Xi = Xq(labels==i, :);
        if isempty(Xi)
            s(i) = 0;
        else
            d = sqrt(sum((Xi - Cx(i,:)).^2, 2));
            s(i) = mean(d);
        end
    end
    Cdist = squareform(pdist(Cx, 'euclidean'));    % 簇心两两距离
    Cdist(eye(K)==1) = inf;                        % 避免除零
    % 轻微数值保护（不改变排序，仅防除零爆炸）
    Cdist = max(Cdist, 1e-12);
    Rij = (s + s.') ./ Cdist;                      % (s_i+s_j)/||c_i-c_j||
    Rij(eye(K)==1) = -inf;
    Ri  = max(Rij, [], 2, 'omitnan');              % 每簇取最大
    dbi = mean(Ri, 'omitnan');                     % 再求平均
end

function col = excel_col(n)
% 1-based 列号转 Excel 列字母（1->A, 2->B, 27->AA,...）
    assert(n>=1, '列号必须>=1');
    col = "";
    while n > 0
        r = mod((n-1), 26);
        col = char('A' + r) + col;
        n = floor((n-1)/26);
    end
end

function [net_best, W_best, bmu_best, qe_best, te_best, best_seed, te_fixed_best, base, cut] = ...
         train_som_multistart(Xq, x, d1, d2, seed_base, repeats, coverSteps, initNeighbor, epochsFine)
% 多次重启 SOM，返回 TE 最小的网络（含 seed=seed_base）
    qe_best = inf; te_best = inf; best_seed = NaN; te_fixed_best = NaN;
    net_best = []; W_best = []; bmu_best = [];
    history = zeros(repeats, 3);  % [seed, QE, TE_adapt]
    for t = 1:repeats
        seed = seed_base + (t-1); rng(seed,'twister');  % ← 包含 seed_base 本身（如 200）
        net = selforgmap([d1 d2], coverSteps, initNeighbor, 'hextop', 'linkdist');
        net.inputs{1}.processFcns = {};    % 禁用 mapminmax
        net.trainParam.epochs = epochsFine;
        [net, ~] = train(net, x);

        W   = net.IW{1};
        y   = net(x);
        bmu = vec2ind(y);

        % ---- QE ----
        qe = mean( sqrt( sum( (Xq - W(bmu,:)).^2 , 2) ), 'omitnan' );

        % ---- TE（自适应 & 固定阈值调试）----
        pos = net.layers{1}.positions.';
        P   = squareform(pdist(pos));
        Dwx = pdist2(W, Xq, 'euclidean');
        [~, order] = sort(Dwx, 1, 'ascend');
        b1 = order(1, :).';
        b2 = order(2, :).';

        % 固定阈值（旧口径）仅用于调试
        adj_fixed = (P > 0) & (P <= 1.05);
        te_fixed  = mean(arrayfun(@(i,j) ~adj_fixed(i,j), b1, b2));

        nz   = P(P>0);
        base = prctile(nz, 1);
        cut  = 1.10 * base;
        adj_adapt = (P > 0) & (P <= cut);
        te_adapt  = mean(arrayfun(@(i,j) ~adj_adapt(i,j), b1, b2));

        history(t,:) = [seed, qe, te_adapt];
        fprintf('[重启 %2d/%2d] seed=%d | QE=%.6f | TE_adapt=%.6f | TE_fixed=%.6f\n', ...
                t, repeats, seed, qe, te_adapt, te_fixed);

        % 以 TE_adapt 为主，QE 作为次判据
        better = (te_adapt < te_best) || (abs(te_adapt-te_best) < 1e-9 && qe < qe_best);
        if better
            te_best = te_adapt; qe_best = qe; best_seed = seed;
            te_fixed_best = te_fixed;
            net_best = net; W_best = W; bmu_best = bmu;
        end
    end

    % 简要汇总
    [~,ix] = min(history(:,3));
    fprintf('—— 多次重启摘要 ——\n');
    fprintf('TE_adapt 最小值 = %.6f（seed=%d）\n', history(ix,3), history(ix,1));
    fprintf('TE_adapt 中位数 = %.6f | 平均值 = %.6f\n', median(history(:,3)), mean(history(:,3)));
end




%% ================== SOM 输入平面（最终版，先画再接管 figure） ==================
% 使用方法：在训练好 net 之后直接运行本段
% 注意：plotsomplanes 会自己创建图窗，所以要在它之后用 gcf 抓到正确的 figure

% ---------------- 配置区 ----------------
colorMapName   = 'hot';       % 'hot' | 'parula' | 'jet' | ...
titleText      = '颜色表示该输入变量与各神经元连接权重的大小（亮色=高，暗色=低）';

% 色轴模式：
% 'auto_minmax'     = [min(W), max(W)]（默认，直观）
% 'auto_symmetric'  = [-maxAbs, +maxAbs]（强调正负对称）
% 'manual'          = 手动指定 climManual = [a b]
% 'percentile'      = 百分位裁剪（去掉极端值），例如 [5, 95]
climMode       = 'auto_minmax';
climManual     = [0 1];           % 仅当 climMode='manual' 时使用
climPercentile = [5 95];          % 仅当 climMode='percentile' 时使用
% ----------------------------------------

% 1) 权重范围计算
W = net.IW{1};   % M×D
wmin = min(W(:));
wmax = max(W(:));

switch lower(climMode)
    case 'auto_minmax'
        clim = [wmin, wmax];
    case 'auto_symmetric'
        wabs = max(abs([wmin, wmax]));
        clim = [-wabs, +wabs];
    case 'manual'
        clim = climManual;
    case 'percentile'
        pLo = prctile(W(:), climPercentile(1));
        pHi = prctile(W(:), climPercentile(2));
        if ~isfinite(pLo) || ~isfinite(pHi) || pLo==pHi
            clim = [wmin, wmax];  % 兜底
        else
            clim = [pLo, pHi];
        end
    otherwise
        clim = [wmin, wmax];      % 兜底
end

% 2) 先由 plotsomplanes 生成图（它会自己开一个 figure）
plotsomplanes(net);



% 3) 现在抓到 plotsomplanes 刚刚创建的那个 figure，并接管它
fig = gcf;  % 很关键：获取真正有子图的图窗
set(fig, 'Color','w', 'Name','SOM 输入平面（统一色条）', ...
         'Units','normalized','Position',[0.05 0.05 0.85 0.85]);
colormap(fig, colorMapName);

%% === 用表格中的指标名替换 planes 标题 ===
% 依赖：indicator_names 已按行顺序读出（与你的数据 D×Q 中的 D 一致）

fig = gcf;
ax_all = findobj(fig,'Type','axes');
% 过滤掉色条等非数据轴
ax_planes = ax_all(~arrayfun(@(h) isa(h,'matlab.graphics.illustration.ColorBar'), ax_all));

D = numel(indicator_names);
for h = ax_planes.'
    % 取原始标题，例如 “来自输入 7 的权重”/“From input 7 weights”
    oldTitle = get(get(h,'Title'),'String');
    % 抓取标题里最后一个数字 n
    tok = regexp(string(oldTitle), '(\d+)', 'tokens', 'once');
    if ~isempty(tok)
        n = str2double(tok{1});
        if isfinite(n) && n>=1 && n<=D
            % 改成你表格里的名字；不转义特殊字符
            title(h, string(indicator_names(n)), 'Interpreter','none');
        end
    end
end
%% === 标题替换：仅化学名（无单位），自动上下标 ===
% 建议：使用 TeX（或改 'latex'）
set(groot,'defaultTextInterpreter','tex', ...
          'defaultAxesTickLabelInterpreter','tex', ...
          'defaultLegendInterpreter','tex');

% 1) 找到所有平面子图（剔除色条）
ax_all = findobj(gcf,'Type','axes');
ax = ax_all(~arrayfun(@(h) isa(h,'matlab.graphics.illustration.ColorBar'), ax_all));

% 2) 按“从上到下、从左到右”排序，确保与 plotsomplanes 的视觉顺序一致title
pos = cell2mat(get(ax,'Position'));                     % [x y w h]
centers = [pos(:,1)+pos(:,3)/2, pos(:,2)+pos(:,4)/2];
[~,ord] = sortrows([-centers(:,2), centers(:,1)]);      % y降序，x升序
ax_sorted = ax(ord);

% 3) 逐个映射表头名称到标题（无单位）
D = min(numel(ax_sorted), numel(indicator_names));
titleFontSize = 13;     % ← 想要的字号，按需改 14/16/18
for i = 1:D
    nice = fmt_chem_no_unit(string(indicator_names(i)));
    h = title(ax_sorted(i), nice, 'Interpreter','tex', 'FontSize', titleFontSize);
    % 如需更醒目： set(h,'FontWeight','bold');
end

%for i = 1:D
    %nice = fmt_chem_no_unit(string(indicator_names(i)));  % ↓ 本地函数
    %title(ax_sorted(i), nice, 'Interpreter','tex');        % 或 'latex'
%end

%% ---- 本地函数：仅化学名 → 化学式（带上下标），无单位 ----
function t = fmt_chem_no_unit(s)
    s = strtrim(string(s));      % 原始列名（不含单位）
    % 用于匹配的 key：全大写+去空格（仅用于判断，不用于显示）
    key = upper(regexprep(s,'\s+',''));

    switch key
      
        % ===== 阴离子（固定规范写法）=====
        case {'HCO3','HCO_3'} , core = "HCO_3^-";
        case {'CO3','CO_3'}   , core = "CO_3^{2-}";
        case {'SO4','SO_4'}   , core = "SO_4^{2-}";
        case {'NO3','NO_3'}   , core = "NO_3^-";
        case {'CL'}           , core = "Cl^-";
        case {'F'}            , core = "F^-";
        case {'BR'}           , core = "Br^-";
        case {'SIO3','SIO_3'} , core = "SiO_3^{2-}";
        % ===== 阳离子 =====
        case {'NA'}           , core = "Na^+";
        case {'K'}            , core = "K^+";
        case {'CA'}           , core = "Ca^{2+}";
        case {'MG'}           , core = "Mg^{2+}";
        case {'AL'}           , core = "Al^{3+}";
        case {'ZN'}           , core = "Zn^{2+}";
        case {'CU'}           , core = "Cu^{2+}";
        case {'PB'}           , core = "Pb^{2+}";
        % ===== 中性或特殊 =====
        case {'lon'}           , core = "lon";
        case {'lat'}           , core = "lat";
        case {'AS'}           , core = "As";
        case {'SE'}           , core = "Se";
        case {'TDS'}          , core = "TDS";
        case {'PH'}           , core = "pH";
        case {'TH'}           , core = "TH";
        case {'NO3-N','NO3N','NO3_N','NO3AS N','NO3-AS-N'}
                                core = "NO_3^-{-}N";
       % δ18O / δD 常见别名都兼容一下
       case {'Δ18O','DELTA18O','D18O','δ18O','δ^18O'}
           core = "\delta^{18}O";
       case {'ΔD','Δ2H','DELTA2H','D2H','δD','δ2H','δ^2H'}
           core = "\delta^{2}H";
        % ===== 兜底：尝试把常见写法正则成上下标；否则原样 =====
        otherwise
            core = string(s);   % 先按原样
            % 常见离子模式：SO42- / SO4 2- 等 → 上下标
            core = regexprep(core,'(?i)\bSO\s*4\s*2-','SO_4^{2-}');
            core = regexprep(core,'(?i)\bCO\s*3\s*2-','CO_3^{2-}');
            core = regexprep(core,'(?i)\bNO\s*3\s*-','NO_3^-');
            core = regexprep(core,'(?i)\bHCO\s*3\s*-','HCO_3^-');
            % 简单把阳离子“元素+价”转成上标：Ca2+ → Ca^{2+}
            core = regexprep(core,'(?i)\b([A-Z][a-z]?)\s*([0-9])\+','${upper($1)}^{${$2}+}');
            core = regexprep(core,'(?i)\b([A-Z][a-z]?)\s*([0-9])-','${upper($1)}^{${$2}-}');
    end

    t = core;  % 无单位，直接返回
end



%% ===== 固定像素布局：保持子图大小，仅调间距 & 色条偏移（最终稳定版） =====
fig = gcf;                                  % 当前 SOM 输入平面图窗口
set(fig,'Units','pixels');                  % 以像素为基准

% --------- 可调参数（按需修改） ---------
% 1) 子图网格：默认自动；若想手动指定，填正整数即可
numCols = [];     % 例如 5
numRows = [];     % 例如 4

% 2) 子图间距（像素）
hGap   = 0;      % 子图之间的水平间距
vGap   = 55;      % 子图之间的垂直间距

% 3) 图边距（像素）
leftMargin   = 40;
rightMargin  = 90;   % 右侧额外留白（在色条右边）
topMargin    = 40;
bottomMargin = 40;

% 4) 色条设置（像素）
cbWidthPx  = 16;     % 色条宽度
cbHOffset  = 20;     % 色条距右侧子图群的“横向”距离
cbVOffset  = 2.5;      % 色条整体上下微调（+上移 / -下移）
hideCbLabel = true;  % 不需要文字标注 → true
% --------------------------------------

% —— 收集所有子图 axes（排除 colorbar 对象） ——
ax_all = findobj(fig,'Type','axes');
% 有些版本 colorbar 不是 axes，这里稳妥过滤任何 ColorBar 类对象
ax = ax_all(~arrayfun(@(h) isa(h,'matlab.graphics.illustration.ColorBar'), ax_all));
if isempty(ax)
    warning('未找到输入平面的坐标轴(ax)。请确认已调用 plotsomplanes(net) 且当前 figure 正确。');
    return;
end

% —— 一律改为像素坐标 —— 
for k = 1:numel(ax)
    set(ax(k),'Units','pixels');
end

% —— 记录子图“固定尺寸”：用平均宽/高 —— 
posPix = cell2mat(get(ax,'Position'));
tileW  = round(mean(posPix(:,3)));
tileH  = round(mean(posPix(:,4)));

% —— 子图按“上到下、左到右”排序，避免顺序错乱 —— 
centers = [posPix(:,1)+posPix(:,3)/2, posPix(:,2)+posPix(:,4)/2];
[~,ord] = sortrows([-centers(:,2), centers(:,1)]);  % y降序、x升序
ax = ax(ord);

% —— 自动/手动确定网格 —— 
N = numel(ax);
if isempty(numCols) || isempty(numRows)
    numCols_auto = ceil(sqrt(N));
    numRows_auto = ceil(N/numCols_auto);
    if isempty(numCols), numCols = numCols_auto; end
    if isempty(numRows), numRows = numRows_auto; end
end

% —— 计算子图群所需宽高（像素） —— 
blockW = numCols*tileW + (numCols-1)*hGap;
blockH = numRows*tileH + (numRows-1)*vGap;

% —— 初始窗口尺寸 & 左下起点 —— 
figPos = get(fig,'Position');  figW = figPos(3);  figH = figPos(4);
left0   = leftMargin;
bottom0 = bottomMargin;

% —— 预估色条位置（像素），并“先扩窗口”以容纳色条（关键！）——
needW = left0 + blockW + cbHOffset + cbWidthPx + rightMargin;
needH = bottom0 + blockH + topMargin;
if figW < needW || figH < needH
    figPos(3) = max(figW, needW);
    figPos(4) = max(figH, needH);
    set(fig,'Position',figPos);
    drawnow;
    figPos = get(fig,'Position'); figW = figPos(3); figH = figPos(4);
end

% —— 放置子图（固定像素位置，保持大小不变） —— 
topBase = figH - topMargin;              % 顶部基线
for i = 1:N
    r = ceil(i/numCols);                 % 行（1=第一行，顶部）
    c = i - (r-1)*numCols;               % 列（1=最左列）
    left_i   = left0 + (c-1)*(tileW + hGap);
    bottom_i = topBase - r*tileH - (r-1)*vGap;
    set(ax(i),'Position',[left_i, bottom_i, tileW, tileH], ...
              'ActivePositionProperty','position','Units','pixels');
end

%% —— 用真实子图包络框对齐色条（修正错位） ——

% 1) 先删掉旧色条，避免干扰
oldcbs = findobj(fig,'Type','ColorBar');
if ~isempty(oldcbs), delete(oldcbs); end

% 2) 以第一个子图为宿主创建色条（继承 CLim/Colormap 最稳）
cb = colorbar(ax(1), 'eastoutside');
set(cb,'Units','pixels','Visible','on');

% 3) 计算“真实子图包络框”（所有 ax 的 Position 联合）
posNow = cell2mat(get(ax,'Position'));   % 每行: [left bottom width height]
x0 = min(posNow(:,1));
y0 = min(posNow(:,2));
x1 = max(posNow(:,1) + posNow(:,3));
y1 = max(posNow(:,2) + posNow(:,4));
bboxW = x1 - x0;
bboxH = y1 - y0;

% 4) 按包络框精准放置色条（这样就与子图顶部/底部齐平）
cbLeft   = x1 + cbHOffset;        % 子图群最右侧 + 横向偏移
cbBottom = y0 + cbVOffset;        % 子图群最低处 + 竖向微调
cbWidth  = cbWidthPx;
cbHeight = bboxH;

set(cb,'Position', [cbLeft, cbBottom, cbWidth, cbHeight]);

% 5) 可选外观
try, cb.Box = 'off'; catch, end
try, cb.Label.String = ''; catch, end

% 6) 锁定窗口，避免缩放引发重排
set(fig,'Resize','off','SizeChangedFcn',[]);
drawnow;
%% ===== 自适应收放窗口（按真实可视包络：包含刻度/标题/色条刻度） =====
% 收集所有需要纳入版面的对象：子图 + 色条
objs = [ax(:).' findobj(fig,'Type','ColorBar').'];  % 行向量

% 计算包含 TightInset 的外包络
x0 = inf; y0 = inf; x1 = -inf; y1 = -inf;
for h = objs
    set(h,'Units','pixels');                       % 像素坐标
    p  = get(h,'Position');                        % [x y w h]
    ti = [0 0 0 0];
    if isprop(h,'TightInset')
        ti = get(h,'TightInset');                  % [left bottom right top]
    end
    left   = p(1) - ti(1);
    bottom = p(2) - ti(2);
    right  = p(1) + p(3) + ti(3);
    top    = p(2) + p(4) + ti(4);
    x0 = min(x0,left);  y0 = min(y0,bottom);
    x1 = max(x1,right); y1 = max(y1,top);
end
contentW = x1 - x0;
contentH = y1 - y0;

% 目标窗口大小 = 内容包络 + 安全边距（像素）
padL = 20; padR = 20; padT = 20; padB = 20;   % 可按需调小/调大
needW = ceil(contentW + padL + padR);
needH = ceil(contentH + padB + padT);

% 以当前左下角为锚，调整 figure 尺寸（只改宽高，不改子图位置）
figPos = get(fig,'Position');
figPos(3) = needW;
figPos(4) = needH;
set(fig,'Position',figPos);

% 如果内容不是从 (0,0) 开始，整体平移到留白内（防止上/左被裁）
dx = padL - x0;
dy = padB - y0;
if abs(dx)>0.5 || abs(dy)>0.5
    for h = objs
        p = get(h,'Position');
        set(h,'Position',[p(1)+dx, p(2)+dy, p(3), p(4)]);
    end
end

% 最终再锁定窗口，避免用户拖拽后重排
set(fig,'Resize','off','SizeChangedFcn',[]);
drawnow;

%% -------------------- PCA 二维可视化（样本标签，稳健版） --------------------
% 要求：Xq (Q×D)，clusterLabel (Q×1, 取值 1..K0)
Q = size(Xq,1);
if numel(clusterLabel) ~= Q
    error('PCA可视化失败：clusterLabel长度(%d)与样本数Q(%d)不一致。', numel(clusterLabel), Q);
end

% 仅保留有效行
valid = all(isfinite(Xq),2) & isfinite(clusterLabel);
if ~any(valid)
    error('PCA可视化失败：Xq 或 clusterLabel 全是无效值。');
end

% 做 PCA（只用有效行）
[~, score, ~, ~, expl] = pca(Xq(valid,:), 'Rows','complete');

% 颜色
Kplot = max(clusterLabel(valid));
cmap  = lines(max(3,Kplot));

% 强制可见并指定位置大小
set(0,'DefaultFigureVisible','on');
figPCA = figure('Color','w','Name',sprintf('PCA scatter (K=%d)', K0), ...
                'Visible','on','Position',[120 120 900 700]);
ax = axes(figPCA); hold(ax,'on');

% 画图（优先 gscatter，失败则回退普通散点）
try
    gscatter(score(:,1), score(:,2), categorical(clusterLabel(valid)), cmap, '.', 18, 'off');
catch
    sc = scatter(ax, score(:,1), score(:,2), 24, double(clusterLabel(valid)), 'filled');
    colormap(ax, cmap); colorbar(ax);
end

xlabel(ax, sprintf('PC1 (%.1f%%)', expl(1)), 'FontSize',11);
ylabel(ax, sprintf('PC2 (%.1f%%)', expl(2)), 'FontSize',11);
title(ax, sprintf('样本聚类结果（PCA 二维可视化，K=%d）', K0), 'FontSize',12);
axis(ax,'equal'); grid(ax,'on'); box(ax,'on');
drawnow;

% 同时保存一份，便于确认是否生成成功
try
    exportgraphics(figPCA, 'pca_scatter.png', 'Resolution', 200);
catch
    saveas(figPCA, 'pca_scatter.png');
end


%% -------------------- SOM 六边形：仅显示数量 N，点击查看全部样品编号 --------------------
% 依赖变量：net, W, bmuIdx, Xq, sample_names, clusterLabel, K0

% —— 网格与六边形几何 —— 
pos  = net.layers{1}.positions.';        % M×2
Dmat = squareform(pdist(pos)); Dmat(Dmat==0) = inf;
dnn  = median(min(Dmat,[],2));
theta   = (0:6)'*pi/3 + pi/6;            % 平顶
hexUnit = [cos(theta) sin(theta)];
r       = dnn / sqrt(3);                 % 边贴边
M       = size(W,1);

% —— 每个神经元命中样本与众数簇标签（0=无命中）——
hitIds  = accumarray(bmuIdx(:), (1:size(Xq,1)).', [M 1], @(v){v}, {[]});
neuronLabel = zeros(M,1);
for i = 1:M
    ids = hitIds{i};
    if ~isempty(ids)
        neuronLabel(i) = mode(clusterLabel(ids));
    end
end

% —— 颜色参数 —— 
cmap       = lines(K0);
emptyColor = [0.85 0.85 0.85];

% —— 画图（仅显示数量 N）—— 
fig = figure('Color','w','Units','pixels','Position',[100 100 900 1200]); 
hold on;

for i = 1:M
    verts = pos(i,:) + r*hexUnit;
    % 颜色：有命中则按簇着色，无命中灰色
    faceColor = emptyColor;
    if neuronLabel(i) >= 1
        faceColor = cmap(neuronLabel(i),:);
    end

    % 六边形面片；把 i 放到 UserData，便于回调查找
    hp = patch(verts(:,1), verts(:,2), faceColor, ...
        'EdgeColor',[0.35 0.35 0.45], 'LineWidth',0.8, 'FaceAlpha',0.95, ...
        'UserData', i, 'HitTest','on', 'PickableParts','all');

    % —— 仅显示数量 N —— 
    ids = hitIds{i};
    if ~isempty(ids)
        N = numel(ids);
        txt = sprintf('N=%d', N);
        fs = 11;                                % 数字字号，适当大一些
        % 带描边的数字，任何底色都清晰
        %haloText(pos(i,1), pos(i,2), txt, [1 1 1], [0 0 0], fs);
        %标准
        text(pos(i,1), pos(i,2), txt, ...
             'HorizontalAlignment','center', ...
             'VerticalAlignment','middle', ...
             'FontSize',fs, ...
             'FontName','Times New Roman', ...
             'Color','w', ...      % 白色字体
             'FontWeight','bold', ...
             'Interpreter','none', ...
             'Clipping','on');

    end
end

% axis equal off
% title(sprintf('SOM 六边形（仅显示命中数量，点击查看样品列表；K=%d）', K0));

axis equal off
hTitle = title(sprintf('SOM 六边形（仅显示命中数量，点击查看样品列表；K=%d）', K0));


% —— 图例 —— 
% ---------- Legend：加大、横排、紧贴主图 ----------
hleg = gobjects(K0 + 1, 1);
for k = 1:K0
    hleg(k) = plot(nan, nan, 's', 'MarkerFaceColor', cmap(k,:), ...
                   'MarkerEdgeColor','none', 'MarkerSize', 10);   % ← 标识块更大
end
hleg(K0+1) = plot(nan, nan, 's', 'MarkerFaceColor', emptyColor, ...
                  'MarkerEdgeColor','none', 'MarkerSize', 10);

labels = [arrayfun(@(k) sprintf('Cluster %d',k), 1:K0, 'uni',0), {'No hits'}];
lgd = legend(hleg, labels{:}, ...
             'Orientation','horizontal', ...
             'Location','southoutside', ...
             'Box','off');

% ① 放大图例字号 & 标识块
lgd.FontSize = 13;            % ← 调大字号（可用 12~14）
lgd.ItemTokenSize = [28, 14]; % ← 调大色块/标记长度和高度

% ② 把图例“贴近”主图：手动定位在坐标轴正下方，间隙很小
ax = gca;
ax.PositionConstraint = 'innerposition';  % 防止 legend 改变坐标轴大小
set(lgd,'Units','normalized');            % 统一用归一化坐标
set(ax,'Units','normalized');

axPos = ax.Position;      % [left bottom width height]
lgdPos = lgd.Position;    % 初始大小

gap = 0.01;  % 与主图的竖向间距（越小越靠近，可调 0.005~0.02）
lgd.Position = [ ...
    axPos(1) + (axPos(3) - lgdPos(3))/2, ...   % 居中到主图正下方
    axPos(2) - lgdPos(4) - gap,       ...      % 紧贴主图下侧
    lgdPos(3), lgdPos(4) ...
];

% （可选）略收紧四周留白
outerPad = 0.02;
set(ax,'LooseInset', max(get(ax,'TightInset'), outerPad*[1 1 1 1]));


% hleg = gobjects(K0 + 1, 1);
% for k = 1:K0
%     hleg(k) = plot(nan, nan, 's', 'MarkerFaceColor', cmap(k,:), ...
%                    'MarkerEdgeColor','none', 'MarkerSize', 8);
% end
% hleg(K0+1) = plot(nan, nan, 's', 'MarkerFaceColor', emptyColor, ...
%                   'MarkerEdgeColor','none', 'MarkerSize', 8);
% legend(hleg, [arrayfun(@(k) sprintf('Cluster %d',k), 1:K0, 'uni',0), {'No hits'}], ...
%        'Location','eastoutside');

% —— 交互：数据光标显示完整样品编号 —— 
setappdata(fig,'hitIds',hitIds);
setappdata(fig,'sample_names',sample_names);
dcm = datacursormode(fig);
set(dcm,'Enable','on','UpdateFcn',@(obj,evt) tipFcn(evt));

% ==========================✅ 在这里插入导出逻辑 ==========================
% 临时隐藏标题
set(hTitle,'Visible','off');

% 自动导出（当前路径或指定路径）
exportgraphics(gcf,'D:\Desktop\SOM-KM\SOM-Kmean\图片导出\som_hex_counts.jpg','Resolution',600);

% 导出完后（可选）恢复标题可见性
set(hTitle,'Visible','on');

%% ==================== 本地函数 ====================

function haloText(x,y,txt,fg,bg,fs)
% 在 (x,y) 绘制带描边文本（先铺 8 向背景，再画前景）
    ax = axis;
    offs = 0.008;                         % 0.006~0.012 可微调
    dx = offs*(ax(2)-ax(1));
    dy = offs*(ax(4)-ax(3));
    for u = -1:1
        for v = -1:1
            if u==0 && v==0, continue; end
            text(x+u*dx, y+v*dy, txt, ...
                'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
                'FontSize',fs, 'Color', bg, 'Interpreter','none', 'Clipping','on');
        end
    end
    text(x, y, txt, 'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
        'FontSize',fs, 'Color', fg, 'FontWeight','bold', ...
        'Interpreter','none', 'Clipping','on');
end

function out = tipFcn(evt)
% Data Cursor：显示该六边形完整样品编号列表
    fig = ancestor(evt.Target,'figure');
    hitIds = getappdata(fig,'hitIds');
    sample_names = getappdata(fig,'sample_names');

    % 既支持点到 patch，也支持点到文本；尽量从最近 patch 取 i
    h = evt.Target;
    i = get(h,'UserData');
    if isempty(i)
        % 若点到的是 text，对应位置找最近神经元
        p = evt.Position;
        posAll = findobj(fig,'Type','patch');
        if ~isempty(posAll)
            % 使用第一个 patch 的父轴坐标
            % 这里简单地通过 figure 中保存的顺序估计 i
            % （如果你希望更严谨，可将 pos 保存到 appdata 里再最近邻）
        end
        i = 1; % 兜底
    end

    if i<1 || i>numel(hitIds)
        out = {'No hits'}; return;
    end

    ids = hitIds{i};
    if isempty(ids)
        out = {'No hits'}; return; 
    end

    names = string(sample_names(ids));
    % 长列表分成多行展示
    maxPerLine = 6;
    nLine = ceil(numel(names)/maxPerLine);
    lines = strings(1,nLine);
    for k = 1:nLine
        j1 = (k-1)*maxPerLine + 1;
        j2 = min(k*maxPerLine, numel(names));
        lines(k) = strjoin(names(j1:j2), ', ');
    end
    out = [{'Samples:'}; cellstr(lines(:))];
end

