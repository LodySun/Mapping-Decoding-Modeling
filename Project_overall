%% 
% Step1: Load NIfTI data
data = niftiread('/Users/lodysun/Desktop/25SS/3rd period/MDMB/Project/Project1/subj1/bold.nii');
[num_x, num_y, num_z, num_time_points] = size(data);
fmri_data = reshape(data, [], num_time_points)'; % Time points x Voxels

% Load labels file
labels_file = '/Users/lodysun/Desktop/25SS/3rd period/MDMB/Project/Project1/subj1/labels.txt';
labels_data = readtable(labels_file, 'Delimiter', ' ', 'ReadVariableNames', false);
stimuli_conditions = labels_data.Var1;
condition_mapping = {'rest', 'scissors', 'face', 'cat', 'shoe', 'house', 'scrambledpix', 'bottle', 'chair'};

% Every kind of stimuli matches one number
[unique_conditions, ~, condition_indices] = unique(stimuli_conditions);

% Condition mapping
fprintf('Condition Mapping:\n');
for i = 1:length(unique_conditions)
    fprintf('%s -> %d\n', unique_conditions{i}, i);
end

%% 
% Step2: Construct the Design Matrix
num_conditions = length(unique_conditions);
design_matrix = zeros(num_time_points, num_conditions);

% Assign each condition to the corresponding column
for t = 1:num_time_points
    condition_id = condition_indices(t);
    design_matrix(t, condition_id) = 1;
end

% Design Matrix Visualization
figure;
imagesc(design_matrix);
xlabel('Stimulus Types');
ylabel('Time Points');
title('Raw Design Matrix');
colorbar;
xticks(1:length(unique_conditions));
xticklabels(unique_conditions);
xtickangle(45);

%% 
% Step3: Get spatial and temporal resolution
nii_info = niftiinfo('/Users/lodysun/Desktop/25SS/3rd period/MDMB/Project/Project1/subj1/bold.nii');
voxel_size = nii_info.PixelDimensions;
TR = nii_info.raw.pixdim(5);
fprintf('Voxel size: %.2f x %.2f x %.2f mm\n', voxel_size);
fprintf('TR: %.2f seconds\n', TR);

%% 
% Step4: HRF and Convolution
% remove 'rest'
condition_indices_no_rest = condition_indices;
rest_idx = find(strcmp(unique_conditions, 'rest'));
design_matrix_no_rest = design_matrix(:, setdiff(1:size(design_matrix,2), rest_idx));

% Load pre-defined HRF from file
hrf_data = load('/Users/lodysun/Desktop/25SS/3rd period/MDMB/Project/hrf.mat');
disp('Available fields in hrf_data:');
disp(fieldnames(hrf_data));
hrf = hrf_data.hrf_sampled;
disp(['Loaded HRF size: ', num2str(size(hrf))]);

% Convolve each column with HRF
convolved_design_matrix = zeros(size(design_matrix_no_rest));
for c = 1:size(design_matrix_no_rest, 2)
    conv_result = conv(design_matrix_no_rest(:, c), hrf, 'full');
    convolved_design_matrix(:, c) = conv_result(1:size(design_matrix_no_rest,1));
end

% Add constant terms for each run (121 time points per run)
constants = []; 
for ind = 1:12 
    constants = [constants; zeros(1,(ind-1)*121) ones(1,121) zeros(1,1452-ind*121)]; 
end 

% Combine stimulus regressors with constants
final_design_matrix = [convolved_design_matrix constants']; 

% Visualize final design matrix
figure;
imagesc(final_design_matrix);
xlabel('Regressors');
ylabel('Time Points');
title('Final Design Matrix with Run-specific Constants');
colorbar;

% Update condition labels for visualization
conditions_no_rest = unique_conditions(~strcmp(unique_conditions, 'rest'));
if ~iscell(conditions_no_rest)
    conditions_no_rest = cellstr(conditions_no_rest);
end
constant_labels = arrayfun(@(x) sprintf('Run%d', x), 1:12, 'UniformOutput', false);
xticks(1:size(final_design_matrix,2));
xticklabels([conditions_no_rest(:)', constant_labels]);
xtickangle(45);

%% 
% Step5: GLM Analysis
slice_z = round(num_z/2);
slice_data = data(:,:,slice_z,:);
slice_voxels = reshape(slice_data, [], num_time_points)';

% Make sure data type is double
slice_voxels = double(slice_voxels);
final_design_matrix = double(final_design_matrix);

% print to make sure it fits
disp(['Size of slice_voxels: ', num2str(size(slice_voxels))]);
disp(['Size of final_design_matrix: ', num2str(size(final_design_matrix))]);

% Dimension match
if size(slice_voxels, 1) ~= size(final_design_matrix, 1)
    error('Dimension mismatch: Number of time points does not match between data and design matrix');
end

% GLM
B = zeros(size(final_design_matrix, 2), size(slice_voxels, 2));
t_stats = zeros(size(final_design_matrix, 2), size(slice_voxels, 2));

for voxel = 1:size(slice_voxels, 2)
    y = slice_voxels(:, voxel);
    if length(y) > size(final_design_matrix, 1)
        y = y(1:size(final_design_matrix, 1));
    end
    
    [b, ~, residuals, ~, stats] = regress(y, final_design_matrix);
    B(:, voxel) = b;
    
    % t value
    dfe = length(y) - rank(final_design_matrix);  % df
    mse = sum(residuals.^2)/dfe;  % mse
    
    % se matrix
    Xinv = pinv(final_design_matrix);
    se = sqrt(diag(Xinv*Xinv')*mse);  % se
    
    % t stat
    t_stats(:, voxel) = b./se;
end

% Visualization
% interested condition
condition_of_interest = 1;
B_map = reshape(B(condition_of_interest,:), [num_x, num_y]);
t_map = reshape(t_stats(condition_of_interest,:), [num_x, num_y]);

figure;
subplot(1, 2, 1);
imagesc(B_map);
colorbar;
title('Beta Map for First Regressor (Single Slice)');
axis image;
colormap('jet');

subplot(1, 2, 2);
imagesc(t_map);
colorbar;
title('T-Map for First Regressor (Single Slice)');
axis image;
colormap('jet');

% t-map threshold
figure;
threshold = 2;  % threshold
t_map_thresholded = t_map;
t_map_thresholded(abs(t_map) < threshold) = 0;
imagesc(t_map_thresholded);
colorbar;
title(sprintf('Thresholded T-Map (|t| > %g)', threshold));
axis image;
colormap('jet');

%% 
% Step6: Check the alignment of Convolved and Raw
target_condition = 'cat';
target_index = find(strcmp(unique_conditions, target_condition));

if isempty(target_index)
    error('Target condition not found in the design matrix.');
end

raw_signal = design_matrix(:, target_index);
convolved_signal = conv(raw_signal, hrf, 'same');
time_points = (1:length(raw_signal)) * TR;

figure;
plot(time_points, raw_signal, 'b-', 'LineWidth', 2);
hold on;
plot(time_points, convolved_signal, 'r-', 'LineWidth', 2);
hold off;
legend({'Raw Signal (cat)', 'Convolved Signal (cat)'}, 'Location', 'best');
xlabel('Time (s)');
ylabel('Signal Amplitude');
title(['Time Series Comparison for Condition: ', target_condition]);
grid on;

%% 
% Step7: ROI Analysis
% Convert fMRI data to double and Z-score
fmri_data = double(fmri_data);
fmri_data = zscore(fmri_data);

% Load ROI masks
mask_vt = niftiread('/Users/lodysun/Desktop/25SS/3rd period/MDMB/Project/Project1/subj1/mask4_vt.nii.gz');
mask_face = niftiread('/Users/lodysun/Desktop/25SS/3rd period/MDMB/Project/Project1/subj1/mask8_face_vt.nii.gz');
mask_house = niftiread('/Users/lodysun/Desktop/25SS/3rd period/MDMB/Project/Project1/subj1/mask8_house_vt.nii.gz');

% Convert masks to double and extract voxel indices
mask_vt_voxels = find(double(mask_vt(:)) > 0);
mask_face_voxels = find(double(mask_face(:)) > 0);
mask_house_voxels = find(double(mask_house(:)) > 0);

% Extract BOLD signals for each ROI
time_series_vt = fmri_data(:, mask_vt_voxels);
time_series_face = fmri_data(:, mask_face_voxels);
time_series_house = fmri_data(:, mask_house_voxels);

% Compute Beta coefficients for each ROI
B_vt = pinv(design_matrix) * time_series_vt;
B_face = pinv(design_matrix) * time_series_face;
B_house = pinv(design_matrix) * time_series_house;

% Calculate mean responses and standard errors
mean_response_vt = mean(B_vt, 2);
mean_response_face = mean(B_face, 2);
mean_response_house = mean(B_house, 2);

se_vt = std(B_vt, [], 2) / sqrt(size(B_vt, 2));
se_face = std(B_face, [], 2) / sqrt(size(B_face, 2));
se_house = std(B_house, [], 2) / sqrt(size(B_house, 2));

% Plot results with error bars
figure;
subplot(1,3,1);
bar(mean_response_vt);
hold on;
errorbar(1:length(mean_response_vt), mean_response_vt, se_vt, 'k.', 'LineWidth', 2);
title('VT ROI Responses with SE');
set(gca, 'XTick', 1:length(condition_mapping), 'XTickLabel', condition_mapping);
xtickangle(45);
ylabel('Mean Response ± SE');

subplot(1,3,2);
bar(mean_response_face);
hold on;
errorbar(1:length(mean_response_face), mean_response_face, se_face, 'k.', 'LineWidth', 2);
title('Face ROI Responses with SE');
set(gca, 'XTick', 1:length(condition_mapping), 'XTickLabel', condition_mapping);
xtickangle(45);
ylabel('Mean Response ± SE');

subplot(1,3,3);
bar(mean_response_house);
hold on;
errorbar(1:length(mean_response_house), mean_response_house, se_house, 'k.', 'LineWidth', 2);
title('House ROI Responses with SE');
set(gca, 'XTick', 1:length(condition_mapping), 'XTickLabel', condition_mapping);
xtickangle(45);
ylabel('Mean Response ± SE');

%% 
% Step8: Classification Analysis

% even & odd
runs = reshape(1:12, [], 1);  % runs
odd_runs = runs(1:2:end);     
even_runs = runs(2:2:end);    

% 121 time points per run
time_per_run = 121;
odd_indices = [];
even_indices = [];
for run = 1:12
    run_indices = ((run-1)*time_per_run + 1):(run*time_per_run);
    if ismember(run, odd_runs)
        odd_indices = [odd_indices, run_indices];
    else
        even_indices = [even_indices, run_indices];
    end
end

% Design matrices
X_odd = final_design_matrix(odd_indices, 1:8); 
X_even = final_design_matrix(even_indices, 1:8);

% ROI analysis
rois = {time_series_vt, time_series_face, time_series_house};
roi_names = {'VT', 'Face', 'House'};
classification_results = struct();

% All roi matrices
figure('Position', [100 100 1200 400]);
subplot(1,3,1);

for roi_idx = 1:length(rois)
    roi_data = rois{roi_idx};
    roi_name = roi_names{roi_idx};
    
    % Separate odd and even
    Y_odd = roi_data(odd_indices, :);
    Y_even = roi_data(even_indices, :);
    
    % beta maps
    B_odd = pinv(X_odd) * Y_odd; 
    B_even = pinv(X_even) * Y_even;
    
    % correlation
    correlation_matrix = zeros(8, 8);
    for i = 1:8
        for j = 1:8
            correlation_matrix(i,j) = corr(B_odd(i,:)', B_even(j,:)');
        end
    end
    
    % within-category & between-category
    within_corr = diag(correlation_matrix)';  
    between_corr = correlation_matrix(~eye(size(correlation_matrix)));  
    between_corr = between_corr(:)'; 
    classification_results.(roi_name).correlation_matrix = correlation_matrix;
    classification_results.(roi_name).within_corr = within_corr;
    classification_results.(roi_name).between_corr = between_corr;
    
    % matrices presentation
    subplot(1,3,roi_idx);
    imagesc(correlation_matrix);
    colorbar;
    title([roi_name ' ROI Pattern Correlations']);
    xlabel('Testing Conditions');
    ylabel('Training Conditions');
    conditions_no_rest = unique_conditions(~strcmp(unique_conditions, 'rest'));
    set(gca, 'XTick', 1:length(conditions_no_rest), 'XTickLabel', conditions_no_rest);
    set(gca, 'YTick', 1:length(conditions_no_rest), 'YTickLabel', conditions_no_rest);
    xtickangle(45);
    
    % statistical test
    [h,p] = ttest2(within_corr(:), between_corr(:));
    classification_results.(roi_name).ttest_p = p;
end

figure('Position', [100 100 1200 400]);
subplot(1,1,1);
all_within = [];
all_between = [];
roi_labels_within = {};
roi_labels_between = {};

% All ROI
for roi_idx = 1:length(rois)
    roi_name = roi_names{roi_idx};
    within_corr = classification_results.(roi_name).within_corr;
    between_corr = classification_results.(roi_name).between_corr;
    
    all_within = [all_within; within_corr(:)];
    all_between = [all_between; between_corr(:)];
    
    roi_labels_within = [roi_labels_within; repmat({[roi_name ' Within']}, length(within_corr(:)), 1)];
    roi_labels_between = [roi_labels_between; repmat({[roi_name ' Between']}, length(between_corr(:)), 1)];
end

% All data and labels
all_data = [all_within; all_between];
all_labels = [roi_labels_within; roi_labels_between];

% Boxplot
boxplot(all_data, all_labels);
title('Classification Performance Across ROIs');
ylabel('Correlation');
xtickangle(45);

%% 
% Step9: RDM Analysis

% Extract the activation pattern
rdm_results = struct();

% All ROI's RDM
figure('Position', [100 100 1500 400]);

for roi_idx = 1:length(rois)
    roi_data = rois{roi_idx};
    roi_name = roi_names{roi_idx};
    
    % separate odd and even
    Y_odd = roi_data(odd_indices, :);
    Y_even = roi_data(even_indices, :);
    
    % beta maps
    B_odd = pinv(X_odd) * Y_odd;
    B_even = pinv(X_even) * Y_even;
    
    % construction of RDM
    % pdist & squareform - dissimilarity
    rdm_odd = squareform(pdist(B_odd, 'correlation'));
    rdm_even = squareform(pdist(B_even, 'correlation'));
    rdm_results.(roi_name).rdm_odd = rdm_odd;
    rdm_results.(roi_name).rdm_even = rdm_even;
    
    % RDM Visualization
    subplot(1,3,roi_idx);
    imagesc(rdm_odd);
    colormap('magma');
    colorbar;
    title([roi_name ' ROI RDM (Odd runs)']);
    xlabel('Conditions');
    ylabel('Conditions');
    conditions_no_rest = unique_conditions(~strcmp(unique_conditions, 'rest'));
    set(gca, 'XTick', 1:length(conditions_no_rest), 'XTickLabel', conditions_no_rest);
    set(gca, 'YTick', 1:length(conditions_no_rest), 'YTickLabel', conditions_no_rest);
    xtickangle(45);
end

% Comparing RDM similarity
figure('Position', [100 100 800 600]);
roi_similarity = zeros(length(rois));
for i = 1:length(rois)
    for j = 1:length(rois)
        roi1_rdm = rdm_results.(roi_names{i}).rdm_odd(:);
        roi2_rdm = rdm_results.(roi_names{j}).rdm_odd(:);
        roi_similarity(i,j) = corr(roi1_rdm, roi2_rdm);
    end
end

% Similarity matrix
subplot(2,2,1);
imagesc(roi_similarity);
colorbar;
title('ROI RDM Similarity');
set(gca, 'XTick', 1:length(roi_names), 'XTickLabel', roi_names);
set(gca, 'YTick', 1:length(roi_names), 'YTickLabel', roi_names);
xtickangle(45);

% MDS for visualization
subplot(2,2,2);
hold on;
colors = {'r', 'b', 'g'}; 
legend_entries = {};

for roi_idx = 1:length(rois)
    roi_name = roi_names{roi_idx};
    rdm = rdm_results.(roi_name).rdm_odd;
    
    rdm = (rdm + rdm')/2;  % RDM
    rdm(isnan(rdm)) = 0;   % NaN
    
    % MDS
    try
        mds_coords = mdscale(rdm, 2, 'Criterion', 'metricstress');
        
        % MDS results
        scatter(mds_coords(:,1), mds_coords(:,2), 100, colors{roi_idx}, 'filled');
        text(mds_coords(:,1), mds_coords(:,2), conditions_no_rest, 'FontSize', 8);
        
        legend_entries{end+1} = roi_name;
    catch e
        warning('MDS failed for %s ROI: %s', roi_name, e.message);
    end
end

hold off;
title('MDS Visualization of Conditions');
legend(legend_entries);
xlabel('Dimension 1');
ylabel('Dimension 2');

% Reliability
subplot(2,2,3);
reliability = zeros(1, length(rois));
for roi_idx = 1:length(rois)
    roi_name = roi_names{roi_idx};
    rdm_odd = rdm_results.(roi_name).rdm_odd(:);
    rdm_even = rdm_results.(roi_name).rdm_even(:);
    reliability(roi_idx) = corr(rdm_odd, rdm_even);
end

bar(reliability);
set(gca, 'XTick', 1:length(roi_names), 'XTickLabel', roi_names);
title('RDM Reliability (odd-even correlation)');
ylabel('Correlation');
xtickangle(45);

% Exemplar Discriminability Index
subplot(2,2,4);
edi = zeros(1, length(rois));
for roi_idx = 1:length(rois)
    roi_name = roi_names{roi_idx};
    rdm_odd = rdm_results.(roi_name).rdm_odd;
    rdm_even = rdm_results.(roi_name).rdm_even;
    
    % diagonal & non-diagonal
    diagonal = diag(rdm_odd);
    off_diagonal = rdm_odd(~eye(size(rdm_odd)));
    
    % EDI calculaton
    edi(roi_idx) = mean(diagonal) - mean(off_diagonal);
end

bar(edi);
set(gca, 'XTick', 1:length(roi_names), 'XTickLabel', roi_names);
title('Exemplar Discriminability Index');
ylabel('EDI');
xtickangle(45);
