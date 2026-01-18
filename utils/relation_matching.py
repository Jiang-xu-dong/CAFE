import copy
import json
import pickle
import sys
import pycocotools.mask as mask_utils
import numpy as np
from itertools import groupby
from collections import Counter
import os
import os
from pathlib import Path
from PIL import Image
import cv2

class PVSGRelationAnnotation:
    def __init__(self, anno_file, split='train'):
        with open(anno_file, 'r') as f:
            anno = json.load(f)

        self.video_ids = []
        for data_source in [ 'nsfw']:
            for video_id in anno['split'][data_source][split]:
                self.video_ids.append(video_id)

        self.classes = anno['objects']['thing'] + anno['objects']['stuff']
        self.relations = anno['relations']

        self.videos = {}
        for video_anno in anno['data']:
            self.videos[video_anno['video_id']] = video_anno

    def __getitem__(self, vid):
        assert vid in self.videos
        video_info = copy.deepcopy(self.videos[vid])
        num_frames= video_info['meta']['num_frames']
        object_list, relation_list = [], []
        for object_content in video_info['objects']:
            object_content['category'] = self.classes.index(
                object_content['category'])
            object_list.append(object_content)

        for relation_content in video_info['relations']:
            if relation_content[2] in self.relations:
                relation_content[2] = self.relations.index(relation_content[2])
                relation_list.append(relation_content)

        return {
            'video_id': vid,
            "num_frames": num_frames,
            'objects': object_list,
            'relations': relation_list,
            'relation_str': self.videos[vid]['relations']
        }


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(filepath, data):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def reassign_tids_with_gap(results,start_change_frame_id):

    new_results = []
    person_tid=0
    for r in results:
        if r['cid']=='0':
            person_tid=int(r['tid'])
            break
    for r in results:
        tid = int(r['tid'])
        fid = int(r['fid'])
        cid= int(r['cid'])
        if cid==0:
            if fid < start_change_frame_id:
                new_results.append(r)
                continue
            if fid>=start_change_frame_id:
                r['tid']=str(person_tid+1)
                new_results.append(r)
  
        else :
            if tid<person_tid:
                new_results.append(r)
            if tid> person_tid:
                r['tid']=str(tid+1)
                new_results.append(r)

    return new_results,person_tid





def get_pred_mask_tubes_one_video(vid, work_dir):
    labels = []
    results = []

    # Read mask labels from the file
    label_path = f'/root/openpvsg/task2_30_frames/{vid}/quantitive/masks.txt'
    with open(label_path, 'r') as f:
        for line in f:
            labels.append(line.strip().split())
    max_tid=0
    cid_all=set()
    # Decode mask labels
    for label in labels:
        frame_id, track_id, cid, h, w, m = label
        cid_all.add(cid)
        rle = {'size': (int(h), int(w)), 'counts': m}
        max_tid=max(max_tid,int(track_id))
        mask = mask_utils.decode(rle)
        results.append(dict(fid=frame_id, tid=track_id, mask=mask, cid=cid))

    # Sort data by 'tid' key
    def key_func(k):
        return k['tid']
    
    old_frame_id=0
    start_change_frame_id=0
    count=0
    flag=0
    idx=0
    results = sorted(results, key=key_func)
    
    for r in results:

        frame_id=r['fid']     
        cha=int(frame_id)-old_frame_id
        
        if r['cid'] == '0' and cha>2 and flag==0 and cha not in (34,41,49,50):
            start_change_frame_id=int(frame_id)
            count+=1
            if count==2:
                flag==1
        if r['cid']=='0':
            old_frame_id=int(frame_id)
        else: 
            old_frame_id=1000
    if count!=0:
        results,idx= reassign_tids_with_gap(results,start_change_frame_id)
    # Group by tid
    masks_grp_by_tid = {}
    for key, value in groupby(results, key_func):
        masks_grp_by_tid[key] = list(value)

    # Organize masks into tubes
    pred_mask_tubes = {}
    for key in masks_grp_by_tid.keys():
        class_ids = []
        mask_list = []
        for content in masks_grp_by_tid[key]:
            mask_list.append({int(content['fid']) - 1: content['mask']})
            class_ids.append(content['cid'])
        count = Counter(class_ids)
        tube_class, _ = count.most_common(1)[0]
        pred_mask_tubes[int(key)] = {'cid': tube_class, 'mask': mask_list}

    return pred_mask_tubes,idx


def get_gt_mask_tubes_one_video(vid, pvsg_dataset, data_dir='./data'):
    # find data_source
    data_source='nsfw'
    gt_masks_root_vid = os.path.join(data_dir, data_source, 'masks', vid)
    gt_pan_mask_paths = [
        str(x) for x in sorted(Path(gt_masks_root_vid).rglob('*.png'))
    ]
    object_list = pvsg_dataset[vid]['objects']
    mask_tubes = dict()

    for frame_id, mask_path in enumerate(gt_pan_mask_paths):
        pan_mask = np.array(Image.open(mask_path))
        for object_entry in object_list:
            instance_id = object_entry['object_id']
            if instance_id not in mask_tubes:
                # Initialize the dictionary for instance_id if it doesn't exist
                mask_tubes[instance_id] = {
                    'cid': object_entry['category'],
                    'mask': []
                }
            # No need to set 'cid' every time, it should be the same for each instance_id
            mask_tubes[instance_id]['mask'].append(
                {frame_id: (pan_mask == instance_id).astype(int)})

    return mask_tubes


def convert_to_ranges(frames):
    # this function converts list into a range
    sorted_frames = sorted(frames)
    new_ranges = []
    range_start = sorted_frames[0]
    for i in range(1, len(sorted_frames)):
        if sorted_frames[i] - sorted_frames[i - 1] + 1 > 4:
            if sorted_frames[i - 1] - range_start >= 4:
                new_ranges.append([range_start, sorted_frames[i - 1]])
            range_start = sorted_frames[i]
    if sorted_frames[-1] - range_start >= 4:
        new_ranges.append([range_start, sorted_frames[-1]])

    return new_ranges


def calculate_iou(gt_mask, pred_mask):
    # This is a placeholder function. You should implement the actual IOU calculation here.
    # It should return the Intersection over Union of two masks.
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    if union == 0:
        return 0
    else:
        return intersection / union


def match_tubes(gt_mask_tubes, pred_mask_tubes):
    matching_dict = {}

    # Iterate over each GT tube
    for gt_id, gt_tube in gt_mask_tubes.items():
        gt_cid = gt_tube['cid']
        matching_dict[gt_id] = {}
        # Retrieve all pred tubes with the same cid
        candidate_pred_tubes = {
            pred_id: pred_tube
            for pred_id, pred_tube in pred_mask_tubes.items()
            if int(pred_tube['cid']) == int(gt_cid)
        }
        gt_frames = set(map(lambda x: list(x.keys())[0], gt_tube['mask']))

        # Calculate Tube-IOU for these candidates
        for pred_id, pred_tube in candidate_pred_tubes.items():
            # We need to determine the temporal overlap first
            pred_frames = set(
                map(lambda x: list(x.keys())[0], pred_tube['mask']))
            overlap_frames = gt_frames.intersection(pred_frames)

            # Calculate IOU for overlapping frames and accumulate
            total_iou = 0
            for frame in overlap_frames:
                gt_mask = next(item for item in gt_tube['mask']
                               if list(item.keys())[0] == frame)[frame]
                pred_mask = next(item for item in pred_tube['mask']
                                 if list(item.keys())[0] == frame)[frame]
                total_iou += calculate_iou(gt_mask, pred_mask)
                if calculate_iou(gt_mask, pred_mask) > 0.5:
                    if pred_id not in matching_dict[gt_id]:
                        matching_dict[gt_id][pred_id] = [frame]
                    else:
                        matching_dict[gt_id][pred_id].append(frame)

    return matching_dict


def match_and_process_gt_tubes(vid,
                               pvsg_dataset,
                               pred_mask_tubes,
                               data_dir='./data'):
    # Determine the data source

    data_source='nsfw'

    gt_masks_root_vid = os.path.join(data_dir, data_source, 'masks', vid)

    matching_dict = {}
    object_list = pvsg_dataset[vid]['objects']

    for frame_id, mask_path in enumerate(
            sorted(Path(gt_masks_root_vid).rglob('*.png'))):
        pan_mask = np.array(Image.open(mask_path))

        for object_entry in object_list:
            instance_id = object_entry['object_id']
            cid = object_entry['category']

            # Process the GT mask for this frame
            gt_mask = (pan_mask == instance_id).astype(bool)

            # Prepare candidate prediction masks with the same cid
            candidate_pred_tubes = {
                pred_id: pred_tube
                for pred_id, pred_tube in pred_mask_tubes.items()
                if int(pred_tube['cid']) == int(cid)
            }

            for pred_id, pred_tube in candidate_pred_tubes.items():
                # Find overlapping frames
                pred_frames = set(
                    map(lambda x: list(x.keys())[0], pred_tube['mask']))
                if frame_id in pred_frames:
                    pred_mask = next(
                        item for item in pred_tube['mask']
                        if list(item.keys())[0] == frame_id)[frame_id]
                    pred_mask_resized = cv2.resize(pred_mask.astype('uint8'),
                                   (gt_mask.shape[1], gt_mask.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
                    pred_mask_resized = pred_mask_resized.astype(bool)

                    iou = calculate_iou(gt_mask, pred_mask_resized)

                    if iou > 0.25:
                        if instance_id not in matching_dict:
                            matching_dict[instance_id] = {pred_id: [frame_id]}
                        else:
                            if pred_id not in matching_dict[instance_id]:
                                matching_dict[instance_id][pred_id] = [
                                    frame_id
                                ]
                            else:
                                matching_dict[instance_id][pred_id].append(
                                    frame_id)

    return matching_dict


def find_ranges(num_list):
    ranges = []
    start = num_list[0]
    for i in range(1, len(num_list)):
        if num_list[i] > num_list[i - 1] + 5:
            end = num_list[i - 1]
            ranges.append(f'{start}-{end}')
            start = num_list[i]
    # Add the last range
    ranges.append(f'{start}-{num_list[-1]}')
    return ranges


def compact_matching_dict(matching_dict):
    processed_dict = {}

    for outer_key, inner_dict in matching_dict.items():
        processed_inner = {}
        for inner_key, number_list in inner_dict.items():
            # Rule: Delete the inner key if the list has fewer than 5 numbers
            if len(number_list) < 5:
                continue

            # Rule: If there's only one inner key, convert the list to a range string
            if len(inner_dict) == 1:
                min_val, max_val = min(number_list), max(number_list)
                processed_inner[inner_key] = f'{min_val}-{max_val}'
            else:
                # Sorting the list to ensure continuity
                sorted_num_list = sorted(number_list)
                processed_inner[inner_key] = find_ranges(sorted_num_list)

        if processed_inner:
            processed_dict[outer_key] = processed_inner

    return processed_dict


def translate_gt_relations(matching_dict, gt_relations):
    translated_relations = []

    def time_overlap(range1, range2):
        # This function checks if two ranges overlap and returns the overlapping range
        return [max(range1[0], range2[0]), min(range1[1], range2[1])]

    def is_valid_range(range1):
        # This function checks if the start of the range is less than the end
        return range1[0] < range1[1]

    def merge_sublists(lst):
        merged_list = []
        temp_dict = {}
        for sublist in lst:
            # Extract the key (first three items) and value (fourth item)
            key = tuple(sublist[:-1])
            value = sublist[-1]

            # If key is already in dictionary, append the value to the existing entry
            if key in temp_dict:
                temp_dict[key].append(value)
            else:
                # Otherwise, create a new entry with this value in a list
                temp_dict[key] = [value]

        # Convert the dictionary back into a list with merged items
        for key, values in temp_dict.items():
            merged_list.append(list(key) + [values])

        return merged_list

    for relation in gt_relations:
        tube_1, tube_2, label, time_ranges = relation
        if tube_1 not in matching_dict or tube_2 not in matching_dict:
            continue
        tube_1_ranges = matching_dict[tube_1]
        tube_2_ranges = matching_dict[tube_2]
        for inner_key_1 in tube_1_ranges.keys():
            for inner_key_2 in tube_2_ranges.keys():
                translated_relations.append([inner_key_1, inner_key_2, label, time_ranges])
        # for time_range in time_ranges:
        #     for inner_key_1, ranges_1 in tube_1_ranges.items():
        #         if isinstance(ranges_1, str):  # convert string range to list
        #             ranges_1 = [ranges_1]
        #         for range_str_1 in ranges_1:
        #             start_1, end_1 = map(int, range_str_1.split('-'))
        #             for inner_key_2, ranges_2 in tube_2_ranges.items():
        #                 if isinstance(ranges_2,
        #                               str):  # convert string range to list
        #                     ranges_2 = [ranges_2]
        #                 for range_str_2 in ranges_2:
        #                     start_2, end_2 = map(int, range_str_2.split('-'))
        #                     overlap_1 = time_overlap(time_range,
        #                                              [start_1, end_1 + 1])
        #                     overlap_2 = time_overlap(time_range,
        #                                              [start_2, end_2 + 1])
        #                     overlap_both = time_overlap(overlap_1, overlap_2)
        #                     # Check if there is an overlap and the overlap is valid
        #                     if is_valid_range(overlap_both):
        #                         # Append the overlap, inner keys, and label to the translated relations
        #                         translated_relations.append([
        #                             inner_key_1, inner_key_2, label,
        #                             overlap_both
        #                         ])

    return merge_sublists(translated_relations)


def process_relations(pred_relations, pred_feat_tubes, d=256):
    """Process predicted relations and features to generate a list of
    dictionaries containing relation information and features for each subject-
    object pair across the video frames.

    Parameters:
    - pred_relations: A list of tuples containing relation information
        (subject index, object index, relation type, time span).
    - pred_feat_tubes: A list of lists containing feature information for each frame and each tube.
    - d: The dimensionality of the feature vectors (default is 256).

    Returns:
    - output_list: A list of dictionaries with keys 'relation', 'tube_s', 'tube_o', and 'relation_span'.
    """

    output_list = []

    for item in pred_relations:
        tube_s_index, tube_o_index, relation, time_span = item
        video_length = len(pred_feat_tubes[list(pred_feat_tubes.keys())[0]])

        tube_s_feat, tube_o_feat = np.zeros([video_length,
                                             d]), np.zeros([video_length, d])

        relation_span = np.zeros(video_length)
        for span_range in time_span:
            for i in range(span_range[0], span_range[1]):
                relation_span[i] = 1

        # processing subject feature
        for frame_id in range(video_length):
            if pred_feat_tubes[tube_s_index][frame_id] is not None:
                tube_s_feat[frame_id] = pred_feat_tubes[tube_s_index][
                    frame_id]['query_feat']
            else:
                relation_span[frame_id] = 0

        # processing object feature
        for frame_id in range(video_length):
            if pred_feat_tubes[tube_o_index][frame_id] is not None:
                tube_o_feat[frame_id] = pred_feat_tubes[tube_o_index][
                    frame_id]['query_feat']
            else:
                relation_span[frame_id] = 0

        # ignore those without long relation span
        if sum(relation_span) >= 3:
            output_dict = {
                'relation': relation,
                'tube_s': tube_s_feat,
                'tube_o': tube_o_feat,
                'relation_span': relation_span,
            }

            output_list.append(output_dict)

    return output_list


def process_feats(pred_feat_tubes, d=256):
    video_length = len(pred_feat_tubes[list(pred_feat_tubes.keys())[0]])
    output_list = {}
    for tube_id in pred_feat_tubes.keys():
        new_feat_tube = np.zeros([video_length, d])
        for frame_id in range(video_length):
            if pred_feat_tubes[tube_id][frame_id] is not None:
                new_feat_tube[frame_id] = pred_feat_tubes[tube_id][frame_id][
                    'query_feat']

        output_list[tube_id] = new_feat_tube
    return output_list


def process_pairs(pred_relations):
    pair_list = []
    for relation in pred_relations:
        pair_list.append([relation[0], relation[1]])
    return pair_list

def split_feat_tracks(feats_dict,idx,gap_threshold=3):

    new_feats_dict = {}
    count=0
    start=0
    idx=int(idx)
    lst=[]
    for i in range(0,len(feats_dict[1])-1):
        if feats_dict[idx][i] == None :
            if count==0:
                start=i
            count+=1
        elif feats_dict[idx][i] != None and  count>3:
            
            lst.append((start,count))
            count=0
    new_lst=[]


    if len(new_lst) == 0:

        return feats_dict
    max_item = max(new_lst, key=lambda x: x[1])
    start,count=max_item
    end=start+count-1
    feats_0 = feats_dict[idx]
    L = len(feats_0)

    part1 = feats_0[:start] + [None] * (L - start)
    new_feats_dict[idx] = part1

    part2 = [None] * (end + 1) + feats_0[end+1:]

    part2 = part2[:L]

    for i in range(1, len(feats_dict)+1):
        if i<idx:
            new_feats_dict[i] = feats_dict[i]
        if i >idx:
            new_feats_dict[i+1] = feats_dict[i]
            
            
    

    return new_feats_dict



def process_feats_and_relations(pred_relations, pred_feat_tubes,video_length,idx, d=256):

    output_list = []

    for item in pred_relations:
        tube_s_index, tube_o_index, relation, time_span = item
        # video_length = len(pred_feat_tubes[list(pred_feat_tubes.keys())[0]])

        relation_span = np.zeros(video_length)

        for span_range in time_span:
            for item1 in span_range:
                for i in range(item1[0], item1[1]):
                    relation_span[i] = 1

        if sum(relation_span) >= 3:
            output_dict = {
                'subject_index': tube_s_index,
                'object_index': tube_o_index,
                'relation': relation,
                'relation_span': relation_span,
            }

            output_list.append(output_dict)

    if idx!=0:
        pred_feat_tubes=split_feat_tracks(pred_feat_tubes,idx,gap_threshold=3)

    return {'feats': process_feats(pred_feat_tubes), 'relations': output_list}