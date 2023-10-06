import csv
import audioread
import numpy as np
from matplotlib import pyplot as plt

def get_unique_labels(file_list, label_header):
    label = []
    for selection_table in file_list:
        with open(selection_table) as csvfile:
            csv_reader = csv.DictReader(csvfile, delimiter='\t', quotechar='|')
            for row in csv_reader:
                label.append(row[label_header])
    return list(set(label))


def get_detections_from_selections(file_list, label_header, list_remap_label, model_info, performance_info):
## Read the groundtruth labels and creates a dictionnary detections[file_nb][label_nb][clip_nb]
# We have limited number of labeled files and they are ~1000 clips/file: we create a list the length of num_window/label/file - might be the easiest! (and do the same for BirdNET detections and then compare the tables)

# * OK Get the duration of each associated audio file
# * OK Divide them in num_window = (duration - overlap)/(clip_length - overlap), that will be the total number of scores
# * OK Read each manual annotation file as ground truth
#     - Create one variable vector for each label // 
#     - For each clip & each label mark the detections
#     - If more that 20% of the ground truth is present in a clip, we consider it a detection
    
    file_nb = 0
    detections = {}
   
    for selection_table in file_list:
        begin_time_clip = np.linspace(0, model_info['clip_number'][file_nb]*model_info['clip_length'], 
                                     model_info['clip_number'][file_nb])

        # Create an zeroes array accessible as detections[file_nb][label_nb][clip_nb]
        detections[file_nb] = np.zeros((len(list_remap_label),model_info['clip_number'][file_nb]))

        # Read selection table
        with open(selection_table) as csvfile:
            csv_reader = csv.DictReader(csvfile, delimiter='\t', quotechar='|')
            for row in csv_reader:   
                    # Test if the label exists otherwise skip to next line
                    # This goes through each element of the dictionnary list_remap_label
                    if row[label_header] in list_remap_label:

                        begin_time = float(row['Begin Time (s)'])
                        end_time = float(row['End Time (s)'])
                        label_nb = list_remap_label.index(row[label_header])
                        duration = end_time-begin_time

                        # find the clip that contains begin_time
                        ind_clip_begin_time = np.where(begin_time_clip<=begin_time)

                        # find the clip that contains end_time
                        ind_clip_end_time = np.where(begin_time_clip+model_info['clip_length'] >=end_time)

                        # Easy case: the label is within one clip window
                        if ind_clip_begin_time[0][-1] == ind_clip_end_time[0][0]: 
                            detections[file_nb][label_nb][ind_clip_begin_time[0][-1]] = 1 

                        # Case where label is across several clip windows
                        else:
                            # print('Label: ', begin_time, end_time, label_nb)
                            # print('Index begin time: ', ind_clip_begin_time[0][-1], ' Index end time: ', ind_clip_end_time[0][0])
                            # print('Clip start: ', begin_time_clip[ind_clip_begin_time[0][-1]], ' Clip stop: ', begin_time_clip[ind_clip_end_time[0][0]]+clip_length)

                            # is there more than mini_sig_dur_positive * duration in the first clip
                            if begin_time_clip[ind_clip_end_time[0][0]] - begin_time >= performance_info['mini_sig_dur_positive']*duration: 
                                detections[file_nb][label_nb][ind_clip_begin_time[0][-1]] = 1 
                                # print('Signal marked in clip 1',  (begin_time_clip[ind_clip_end_time[0][0]] - begin_time)/duration)

                            # is there more than mini_sig_dur_positive * duration in the first clip in the last clip
                            if end_time - begin_time_clip[ind_clip_end_time[0][0]] >= performance_info['mini_sig_dur_positive']*duration: 
                                detections[file_nb][label_nb][ind_clip_begin_time[0][0]] = 1
                                # print('Signal marked in clip 2', (end_time - begin_time_clip[ind_clip_end_time[0][0]])/duration)

        file_nb += 1
    # print(begin_time_clip[np.where(detections[0][0]==1)])
    return detections

def get_detections_from_selections_var_threshold(file_list, list_remap_label, model_info, performance_info, threshold = 0):
## Read the model detections and creates a dictionnary of detections[file_nb][label_nb][clip_nb]
# Same as above but allows for a set threshold, to evaluate the preformances
    file_nb = 0
    detections = {}
    label_header = model_info['label_col']
    for selection_table in file_list:
        begin_time_clip = np.linspace(0, model_info['clip_number'][file_nb]*model_info['clip_length'], 
                                      model_info['clip_number'][file_nb])

        # Create an zeroes array accessible as detections[file_nb][label_nb][clip_nb]
        detections[file_nb] = np.zeros((len(list_remap_label),model_info['clip_number'][file_nb]))

        # Read selection table
        with open(selection_table) as csvfile:
            csv_reader = csv.DictReader(csvfile, delimiter='\t', quotechar='|')
            for row in csv_reader:   
                    # Test if the label exists otherwise skip to next line
                    if (row[label_header] in list_remap_label) & (float(row[model_info['score_column']])>=threshold):
                        
                        begin_time = float(row['Begin Time (s)'])
                        end_time = float(row['End Time (s)'])
                        label_nb = list_remap_label.index(row[label_header])
                        duration = end_time-begin_time

                        # find the clip that contains begin_time
                        ind_clip_begin_time = np.where(begin_time_clip<=begin_time)

                        # find the clip that contains end_time
                        ind_clip_end_time = np.where(begin_time_clip+model_info['clip_length'] >=end_time)

                        # Easy case: the label is within one clip window
                        if ind_clip_begin_time[0][-1] == ind_clip_end_time[0][0]: 
                            detections[file_nb][label_nb][ind_clip_begin_time[0][-1]] = 1 

                        # Case where label is across several clip windows
                        else:
                            # print('Label: ', begin_time, end_time, label_nb)
                            # print('Index begin time: ', ind_clip_begin_time[0][-1], ' Index end time: ', ind_clip_end_time[0][0])
                            # print('Clip start: ', begin_time_clip[ind_clip_begin_time[0][-1]], ' Clip stop: ', begin_time_clip[ind_clip_end_time[0][0]]+model_info['clip_length'])

                            # is there more than mini_sig_dur_positive * duration in the first clip
                            if begin_time_clip[ind_clip_end_time[0][0]] - begin_time >= performance_info['mini_sig_dur_positive']*duration: 
                                detections[file_nb][label_nb][ind_clip_begin_time[0][-1]] = 1 
                                # print('Signal marked in clip 1',  (begin_time_clip[ind_clip_end_time[0][0]] - begin_time)/duration)

                            # is there more than mini_sig_dur_positive * duration in the first clip in the last clip
                            if end_time - begin_time_clip[ind_clip_end_time[0][0]] >= performance_info['mini_sig_dur_positive']*duration: 
                                detections[file_nb][label_nb][ind_clip_begin_time[0][0]] = 1
                                # print('Signal marked in clip 2', (end_time - begin_time_clip[ind_clip_end_time[0][0]])/duration)

        file_nb += 1

    # print(begin_time_clip[np.where(detections[0][0]==1)])
    #print(np.shape(detections[0]))
    #plt.plot(detections[0][0])
    #plt.plot(detections[0][1])
    return detections

def scores(detections_groundtruth, detections_model, nb_test_files):
# Compares the results of the groundtruth and model detections[file_nb][label_nb][clip_nb] dictionnaries to count
# true positives (tp), false positives (fp), false negatives (fn) and true negatives (tn)
    label_nb, clip_nb = np.shape(detections_groundtruth[0])
    tp = {}; fp = {}; fn={}; tn={}

    for test_file_ind in range(nb_test_files): # Go through each file
        tp[test_file_ind] = [0]*label_nb; fp[test_file_ind] = [0]*label_nb; 
        fn[test_file_ind] = [0]*label_nb; tn[test_file_ind] = [0]*label_nb
        for label_ind in range(label_nb): # Go through each category
            comparison = detections_groundtruth[test_file_ind][label_ind]*5 - detections_model[test_file_ind][label_ind]
            
            for ii in range(len(comparison)): # Go through each clip
                if comparison[ii] == 0: tn[test_file_ind][label_ind] += 1 # True negative
                elif comparison[ii] == -1: fp[test_file_ind][label_ind] += 1 # False positive 
                elif comparison[ii] == 5: fn[test_file_ind][label_ind] += 1 # False negative
                elif comparison[ii] == 4: tp[test_file_ind][label_ind] += 1 # True positive

    return tp, fp, fn, tn

def get_precision_recall_per_class(tp, fp, fn, nb_test_files, nb_labels):
# gets precision and recall for one threshold
    precision = []
    recall = []
    # Here we're going through each label
    for label_ind in range(nb_labels):
            tp_total = sum([tp[tt][label_ind] for tt in range(nb_test_files)])
            fp_total = sum([fp[tt][label_ind] for tt in range(nb_test_files)])
            fn_total = sum([fn[tt][label_ind] for tt in range(nb_test_files)])
            precision.append(tp_total/(tp_total + fp_total))
            recall.append(tp_total/(tp_total+ fn_total))

    return precision, recall

def evaluate_precision_recall(label_correspondance_map, detector_file_list, groundtruth_file_list, model_info, performance_info, threshold):
# Full pipeline to evaluate the precision & recall
# Can plot the results as:
# 
# for label in list_labels_model:
#     plt.plot(recall[label],precision[label])
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
    
    ## Get the corresponding label lists
    list_labels_groundtruth = list(label_correspondance_map.keys())
    list_labels_model = [label_correspondance_map[kk] for kk in label_correspondance_map]
    
    ## Evaluate groundtruth detections 
    # Checked, the outputs of detection groundtruth are different for both labels print(detections[0][0],detections[0][1])
    detections_groundtruth = get_detections_from_selections(groundtruth_file_list, performance_info['groundtruth_label_col'], 
                                            list_labels_groundtruth, model_info, performance_info)
    
    ## Evaluate the BirdNET detections at varying thresholds to create precision-recall curves
    # Create where the results are going to be stored
    precision = {}
    recall = {}
    for label in list_labels_model:
        precision[label]=[]
        recall[label]=[]
    
    for th in threshold:
        # Evaluate model at this threshold
        detections_model = get_detections_from_selections_var_threshold(detector_file_list, list_labels_model, model_info, performance_info, threshold = th)


        # Count the tp/fp/fn/tn across all test files
        tp, fp, fn, tn = scores(detections_groundtruth, detections_model, len(groundtruth_file_list))
        
        # Get precision and recall for each class
        pr, re = get_precision_recall_per_class(tp, fp, fn, len(groundtruth_file_list), len(list_labels_model))

        indx = 0
        for label in list_labels_model:
            precision[label].append(pr[indx])
            recall[label].append(re[indx])
            indx += 1

    return precision, recall, list_labels_model