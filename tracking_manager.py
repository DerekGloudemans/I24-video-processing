import torch
import numpy as np
import multiprocessing as mp
import os
import queue
from track_sequence import track_sequence

def get_recordings(ingest_session_path):
    recording_names = []
    for item in os.listdir(os.path.join(ingest_session_path,"recording")):
        recording_names.append(item.split(".mp4")[0])
    return recording_names
    
def get_outputs(ingest_session_path):
    recording_names = []
    for item in os.listdir(os.path.join(ingest_session_path,"tracking_outputs")):
        recording_names.append(item.split("_track_outputs")[0])
    return recording_names

def get_in_progress(in_progress):
    in_progress_names = []
    for key in in_progress.keys():
        in_progress_names.append(in_progress[key])
    return in_progress_names    


if __name__ == "__main__":
    ingest_session_path = "/home/worklab/Data/cv/video/ingest_session_00011"
    
    # get GPU list
    g = torch.cuda.device_count()
    device_list = [torch.cuda.device("cuda:{}".format(idx)) for idx in range(g)]    
    
    
    # availability monitor
    available = np.ones(g)
    in_progress = {}
            
    # create shared queue
    manager = mp.Manager()
    ctx = mp.get_context('spawn')
    com_queue = ctx.Queue()
    all_workers = {}
    DONE = False
    
    while not DONE:        
        
        for idx in range(g):
            # initially, start gpu_count processes
            
            if available[idx] == 1:
                in_prog = get_in_progress(in_progress)
                recordings = get_recordings(ingest_session_path)
                done = get_outputs(ingest_session_path)
               
                if len(in_prog) == 0 and len(recordings) == len(done):
                    DONE = True
                    
                
                avail_recording = None
                for item in recordings:
                    if item in in_prog or item in done:
                        continue
                    else:
                        avail_recording = item
                        break
                
                if avail_recording is not None:
                    # assign this recording to this worker
                    
                    in_progress[idx] = avail_recording
                    available[idx] = 0
                    
                    input_file = os.path.join(ingest_session_path,"recording",avail_recording+".mp4")
                    output_directory = os.path.join(ingest_session_path,"tracking_outputs")
                    config_file = "/home/worklab/Documents/derek/I24-video-processing/config/tracker_setup.config"
                    log_file = None
                    args = [input_file, output_directory, config_file,log_file]
                    
                    
                    kwargs = {"device_id":idx, "com_queue":com_queue}
                    worker = ctx.Process(target=track_sequence,args = args, kwargs=kwargs)
                    all_workers[idx] = (worker)
                    all_workers[idx].start()
                    
                    print("Started worker {} on video sequence {}".format(idx,in_progress[idx]))
        
       

        
        # monitor queue for messages that a worker completed its task
        try:
           message = com_queue.get(timeout = 0)            
        except queue.Empty:
            continue
        
        worker_id = int(message.split(" ")[0])
        
        all_workers[worker_id].terminate()
        all_workers[worker_id].join()
        del all_workers[worker_id]
        
        available[worker_id] = 1
        print("worker {} finished video sequence {}".format(worker_id, in_progress[worker_id]))
        del in_progress[worker_id]
        
        
    print("Finished all video sequences")
    for key in all_workers:
        all_workers[key].terminate()
        all_workers[key].join()
        
        
                
