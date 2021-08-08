import os
import sys
import json
import glob
import time
import copy
import random
import shutil
import argparse
import threading

import cv2
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="$ALFWORLD_DATA/json_2.1.1")
parser.add_argument('--save_path', type=str, default="detector/data/")
parser.add_argument('--smooth_nav', dest='smooth_nav', action='store_true')
parser.add_argument('--time_delays', dest='time_delays', action='store_true')
parser.add_argument('--shuffle', dest='shuffle', action='store_true')
parser.add_argument('--num_threads', type=int, default=1)
parser.add_argument('--reward_config', type=str, default='alfworld/agents/config/rewards.json')
parser.add_argument('--pick_few', dest='pick_few', action='store_true')
parser.add_argument('--val', dest='val', action='store_true')
parser.add_argument('--local', dest='local', action='store_true')
parser.add_argument('--imgae_300', dest='imgae_300', action='store_true')
args = parser.parse_args()


if args.local:
    sys.path.append('/Users/soyeonmin/Documents/alfworld_soyeonm/alfworld')
else:
    sys.path.append('/home/root/alfworld')
import alfworld.gen
import alfworld.gen.constants as constants
from alfworld.gen.utils.video_util import VideoSaver
from alfworld.gen.utils.py_util import walklevel
from alfworld.env.thor_env import ThorEnv
import pickle


TRAJ_DATA_JSON_FILENAME = "traj_data.json"
AUGMENTED_TRAJ_DATA_JSON_FILENAME = "augmented_traj_data.json"

IMAGES_FOLDER = "images"
DEPTH_FOLDER = "depths"
HORIZON_FOLDER = "hors"
MASKS_FOLDER = "masks"
META_FOLDER = "meta"

IMAGE_WIDTH = 400
IMAGE_HEIGHT = 400
if args.imgae_300:
    IMAGE_WIDTH = 300
    IMAGE_HEIGHT = 300

render_settings = dict()
render_settings['renderImage'] = True
render_settings['renderDepthImage'] = True
render_settings['renderObjectImage'] = True
render_settings['renderClassImage'] = True

video_saver = VideoSaver()


def get_image_index(save_path):
    return len(glob.glob(save_path + '/*.png'))


def save_image_with_delays(env, action,
                           save_path, direction=constants.BEFORE):
    im_ind = get_image_index(save_path)
    counts = constants.SAVE_FRAME_BEFORE_AND_AFTER_COUNTS[action['action']][direction]
    for i in range(counts):
        save_image(env.last_event, save_path)
        env.noop()
    return im_ind


def save_image(event, save_path):
    # rgb
    rgb_save_path = os.path.join(save_path, IMAGES_FOLDER)
    rgb_image = event.frame[:, :, ::-1]
    depth_save_path = os.path.join(save_path, DEPTH_FOLDER)
    depth_image = event.depth_frame
    horizon_save_path = os.path.join(save_path, HORIZON_FOLDER)
    horizon = event.metadata['agent']['cameraHorizon']

    # masks
    #mask_save_path = os.path.join(save_path, MASKS_FOLDER)
    #mask_image = event.instance_segmentation_frame

    # dump images
    im_ind = get_image_index(rgb_save_path)
    cv2.imwrite(rgb_save_path + '/%09d.png' % im_ind, rgb_image); pickle.dump(depth_image, open(depth_save_path + '/depth' + '%09d.p' % im_ind, 'wb'))
    #cv2.imwrite(mask_save_path + '/%09d.png' % im_ind, mask_image)
    pickle.dump(horizon, open(horizon_save_path + '/horizon' + '%09d.p' % im_ind, 'wb'))
    return im_ind


def save_images_in_events(events, root_dir):
    for event in events:
        save_image(event, root_dir)


def clear_and_create_dir(path):
    # if os.path.exists(path):
    #     shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)


def get_scene_type(scene_num):
    if scene_num < 100:
        return 'kitchen'
    elif scene_num < 300:
        return 'living'
    elif scene_num < 400:
        return 'bedroom'
    else:
        return 'bathroom'


def get_openable_points(traj_data):
    scene_num = traj_data['scene']['scene_num']
    openable_json_file = os.path.join(alfworld.gen.__path__[0], 'layouts/FloorPlan%d-openable.json' % scene_num)
    with open(openable_json_file, 'r') as f:
        openable_points = json.load(f)
    return openable_points


def explore_scene(env, traj_data, root_dir):
    '''
    Use pre-computed openable points from ALFRED to store receptacle locations
    '''
    openable_points = get_openable_points(traj_data)
    agent_height = env.last_event.metadata['agent']['position']['y']
    for recep_id, point in openable_points.items():
        recep_class = recep_id.split("|")[0]
        action = {'action': 'TeleportFull',
                  'x': point[0],
                  'y': agent_height,
                  'z': point[1],
                  'rotateOnTeleport': False,
                  'rotation': point[2],
                  'horizon': point[3]}
        event = env.step(action)
        
        if point[3] >=0:
            print("horizon is ", point[3])
            save_frame(env, event, root_dir)
        event = env.set_horizon(60)
        save_frame(env, event, root_dir)

        for ri in range(2):
            env.step(dict(action="RotateLeft", degrees = "90", forceAction=True)) 
        event = env.set_horizon(0)
        save_frame(env, event, root_dir)


    return len(openable_points)


def get_objects_num_in_frame(env):
    count = 0
    for k, v in env.last_event.instance_masks.items():
        if np.sum(v) >= 25:
            count +=1
    return count
		

def augment_traj(env, json_file, count):
    # load json data
    with open(json_file) as f:
        traj_data = json.load(f)


    # fresh images list
    traj_data['images'] = list()

    # scene setup
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    object_toggles = traj_data['scene']['object_toggles']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']

    # reset
    scene_name = 'FloorPlan%d' % scene_num
    scene_type = get_scene_type(scene_num)
    env.reset(scene_name)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    root_dir = os.path.join(args.save_path, scene_type)

    imgs_dir = os.path.join(root_dir, IMAGES_FOLDER)
    mask_dir = os.path.join(root_dir, MASKS_FOLDER)
    meta_dir = os.path.join(root_dir, META_FOLDER)
    depth_dir = os.path.join(root_dir, DEPTH_FOLDER)
    hor_dir = os.path.join(root_dir, HORIZON_FOLDER)

    clear_and_create_dir(imgs_dir)
    clear_and_create_dir(mask_dir)
    clear_and_create_dir(meta_dir)
    clear_and_create_dir(depth_dir)
    clear_and_create_dir(hor_dir)

    explored_len = 2*explore_scene(env, traj_data, root_dir)/2

    env.step(dict(traj_data['scene']['init_action']))
    # print("Task: %s" % (traj_data['template']['task_desc']))

    # setup task
    env.set_task(traj_data, args, reward_type='dense')
    rewards = []
    prop = int(len(traj_data['plan']['low_actions'])/ explored_len) 
    if prop == 0:
        prop +=1
    print("prop is ", prop)
    for ll_idx, ll_action in enumerate(traj_data['plan']['low_actions']):
        count +=1
        # next cmd under the current hl_action
        cmd = ll_action['api_action']
        hl_action = traj_data['plan']['high_pddl'][ll_action['high_idx']]

        # remove unnecessary keys
        cmd = {k: cmd[k] for k in ['action', 'objectId', 'receptacleObjectId', 'placeStationary', 'forceAction'] if k in cmd}
        np.random.seed(count)
        chosen = np.random.choice(prop)
        
        if chosen == 0:
            event = env.step(cmd)		
            #save_frame(env, event, root_dir)
            cur_hor = env.last_event.metadata['agent']['cameraHorizon']
            #Set horizon to 0
            env.set_horizon(0)
            max_obj_num = -1; keep_ri = None
            for ri in range(4):
                env.step(dict(action="RotateLeft", degrees = "90", forceAction=True)) 
                obj_num = get_objects_num_in_frame(env)
                if obj_num > max_obj_num:
                    keep_ri = ri; max_obj_num = obj_num
                #Get the number of objects inside the frame
            for ri in range(keep_ri+1):
                env.step(dict(action="RotateLeft", degrees = "90", forceAction=True)) 

            hor_idx = np.random.choice(2)
            #for hor in [0,15,30]:
            hor = [0,15][hor_idx]
                #if abs(hor-cur_hor)> 5:
            event = env.set_horizon(hor); save_frame(env, event, root_dir)
            idx = get_image_index(root_dir)
            #print("idx ", idx, " horizon is ", env.last_event.metadata['agent']['cameraHorizon'])
            #Rotate back 

            chosen_45 =  np.choose(4)
            if chosen_45 == 0:
                event = env.set_horizon(45); save_frame(env, event, root_dir)
                idx = get_image_index(root_dir)

            for ri in range(2):
                env.step(dict(action="RotateLeft", degrees = "90", forceAction=True)) 

            hor_idx = np.random.choice(2)
            #for hor in [0,15,30]:
            hor = [0,15][hor_idx]
                #if abs(hor-cur_hor)> 5:
            event = env.set_horizon(hor); save_frame(env, event, root_dir)
            idx = get_image_index(root_dir)

            if chosen_45 == 0:
                event = env.set_horizon(45); save_frame(env, event, root_dir)
                idx = get_image_index(root_dir)

            for ri in range(2):
                env.step(dict(action="RotateLeft", degrees = "90", forceAction=True)) 

            for ri in range(keep_ri+1):
                env.step(dict(action="RotateRight", degrees = "90", forceAction=True))
            env.set_horizon(cur_hor)
        elif "MoveAhead" in cmd['action']:
            event = env.step(cmd)

        elif "Rotate" in cmd['action']:
            event = env.step(cmd)

        elif "Look" in cmd['action']:
            event = env.step(cmd)

        else:
            if np.random.choice(2) ==0:
                new_event = env.step(cmd)
                if env.last_event.metadata['agent']['cameraHorizon'] >=0:
                    save_frame(env, event, root_dir)
                    #save_frame(env, new_event, root_dir)
                    event = new_event
                    for ri in range(2):
                        event = env.step(dict(action="RotateLeft", degrees = "90", forceAction=True)) 
                    save_frame(env, event, root_dir)
                    for ri in range(2):
                        event = env.step(dict(action="RotateLeft", degrees = "90", forceAction=True)) 



        if not event.metadata['lastActionSuccess']:
            raise Exception("Replay Failed: %s" % (env.last_event.metadata['errorMessage']))
    return count


def save_frame(env, event, root_dir):
    im_idx = save_image(event, root_dir)
    # store color to object type dictionary
    color_to_obj_id_type = {}
    all_objects = event.metadata['objects']
    for color, object_id in event.color_to_object_id.items():
        color_to_obj_id_type[str(color)] = object_id
    meta_file = os.path.join(root_dir, META_FOLDER, "%09d.json" % im_idx)
    with open(meta_file, 'w') as f:
        json.dump(color_to_obj_id_type, f)
    # print("Total Size: %s" % im_idx)


def run():
    '''
    replay loop
    '''
    # start THOR env
    env = ThorEnv(player_screen_width=IMAGE_WIDTH,
                  player_screen_height=IMAGE_HEIGHT)

    skipped_files = []
    finished = []
    cache_file = os.path.join(args.save_path, "cache.json")
    count = 0
    while len(traj_list) > 0:
        lock.acquire()
        json_file = traj_list.pop()
        lock.release()

        print ("(%d Left) Augmenting: %s" % (len(traj_list), json_file))
        try:
            count = augment_traj(env, json_file, count)
            finished.append(json_file)
            with open(cache_file, 'w') as f:
                json.dump({'finished': finished}, f)

        except Exception as e:
                import traceback
                traceback.print_exc()
                print ("Error: " + repr(e))
                print ("Skipping " + json_file)
                skipped_files.append(json_file)

    env.stop()
    print("Finished.")

    # skipped files
    if len(skipped_files) > 0:
        print("Skipped Files:")
        print(skipped_files)


traj_list = []
lock = threading.Lock()

# parse arguments



# cache
cache_file = os.path.join(args.save_path, "cache.json")
if os.path.isfile(cache_file):
    with open(cache_file, 'r') as f:
        finished_jsons = json.load(f)
else:
    finished_jsons = {'finished': []}

# make a list of all the traj_data json files
data_path = os.path.expandvars(args.data_path)
if args.val:
    walk = walklevel(data_path, level=2)
    #walk_copy = copy.deepcopy(walklevel(data_path, level=2))
    np.random.seed(0)
    len_walk = 0
    for dir_name, subdir_list, file_list in walk:
        len_walk +=1
    idxes = np.random.choice(len_walk, 10, replace=False)
    print("idxes is ", idxes) 
    #walk = [walk[i] for i in idxes]
    count_walk = 0

    walk_chosen = []
    for dir_name, subdir_list, file_list in walklevel(data_path, level=2):
        count_walk  +=1
        if count_walk in idxes:
            walk_chosen.append((dir_name, subdir_list, file_list))

    print("count is ", count_walk)
    print("len(idxes) is    ", len(idxes))  
    print("walk chosen is ", walk_chosen)
else:
    walk_chosen = walklevel(data_path, level=2)

for dir_name, subdir_list, file_list in walk_chosen:
    if "trial_" in dir_name:
        json_file = os.path.join(dir_name, TRAJ_DATA_JSON_FILENAME)
        if not os.path.isfile(json_file) or json_file in finished_jsons['finished']:
            continue
        traj_list.append(json_file)

# random shuffle
if args.shuffle:
    random.shuffle(traj_list)

# start threads
threads = []
for n in range(args.num_threads):
    thread = threading.Thread(target=run)
    threads.append(thread)
    thread.start()
    time.sleep(1)
