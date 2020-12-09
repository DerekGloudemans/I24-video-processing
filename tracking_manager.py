import torch
import numpy as np
import multiprocessing as mp
import os
import queue
import time
import psutil
import pynvml
import numpy as np

from track_sequence import track_sequence

pynvml.nvmlInit()


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

def write_to_log(log_file,message,show = True):
    """
    All messages passed to this file should be of the form (timestamp, key, message)
        valid Keys - START_PROC_SESSION - start of a processing session
                     END_PROC_SESSION - end of a processing session
                     INFO - general information
                     SYS - CPU, GPU, mem utilization info, etc.
                     WORKER_START - a tracker process has started (name and PID should be given)
                     WORKER_END - a tracker process has finished (name and PID should be given)
                     WORKER_TERM - a tracker process has been terminated after finishing (name and PID should be given)
                     WARNING - parameters outside of normal operating conditions
                     ERROR - an error has been caught 
        timestamp - time.time() 
        message - string
    """
    
    # format time
    milliseconds = str(np.round(message[0],4)).split(".")[1]
    formatted_time = time.strftime('%Y-%m-%d %H:%M:%S.{}', time.localtime(message[0])).format(milliseconds)

    line = "[{}] {}: {} \n".format(formatted_time,message[1],message[2])   
    if show:
        print(line[:-2])
    with open (log_file,"a+") as f:
        f.writelines([line])
        
def log_system(log_file):
    """
    Logs system utilization metrics to log file
    """
    # log cpu util 
    cpu_util = psutil.cpu_percent()
    cpu_util_ind = psutil.cpu_percent(percpu = True)
    ts = time.time()
    key = "SYS"
    message = "CPU util: {}% -- Individual utils: {}".format(cpu_util,cpu_util_ind)
    write_to_log(log_file,(ts,key,message))
    
    # log GPU util and memory
    try:
        deviceCount = pynvml.nvmlDeviceGetCount()
        for idx in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            name = "GPU {}: {}".format(idx,pynvml.nvmlDeviceGetName(handle).decode("utf-8"))
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            fan_util = pynvml.nvmlDeviceGetFanSpeed(handle)
            gpu_util = util.gpu
            mem_util = util.memory
            
            message = "{}: Kernel:{}%  Mem:{}% Fan:{}%".format(name,gpu_util,mem_util,fan_util)
            ts = time.time()
            key = "SYS"
            write_to_log(log_file,(ts,key,message))
            
    except pynvml.NVMLError as error:
        print(error)
    
    # log memory util
    mem_util = psutil.virtual_memory()
    used = round(mem_util.used / 1e+9,2)
    total = round(mem_util.total / 1e+9,2)
    ts = time.time()
    key = "SYS"
    message = "Memory util: {}%  ({}/{}GB)".format(round(used/total*100,2),used,total)
    write_to_log(log_file,(ts,key,message))
    
    last_log_time = time.time()    
    return last_log_time        
    

if __name__ == "__main__":
    
    log_rate = 15
    last_log_time = 0
    ingest_session_path = "/home/worklab/Data/cv/video/ingest_session_00011"
    log_file = os.path.join(ingest_session_path,"logs","cv_tracking_manager.log")

    write_to_log(log_file,(time.time(),"START_PROC_SESSION","Started processing session."))


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
                    args = [input_file, output_directory, config_file,log_file]
                    kwargs = {"device_id":idx, "com_queue":com_queue}
                    
                    worker = ctx.Process(target=track_sequence,args = args, kwargs=kwargs)
                    all_workers[idx] = (worker)
                    all_workers[idx].start()
                    
                    # write log message
                    ts = time.time()
                    key  = "WORKER_START"
                    text = "Manager started worker {} (PID {}) on video sequence {}".format(idx,all_workers[idx].pid,in_progress[idx])
                    write_to_log(log_file,(ts,key,text))
                    
        
       
        # periodically, write device status to log file
        if time.time() - last_log_time > log_rate:
            last_log_time = log_system(log_file)
        
        
        # monitor queue for messages that a worker completed its task
        try:
           message = com_queue.get(timeout = 0)            
        except queue.Empty:
            continue
        
        # write message to log file
        worker_id = message[3]
        message = message[:3]
        write_to_log(log_file,message)
        
        # if message is a finished task, update manager
        key = message[1]
        if key == "WORKER_END":
            worker_pid = all_workers[worker_id].pid
            
            all_workers[worker_id].terminate()
            all_workers[worker_id].join()
            del all_workers[worker_id]
            
            # write log message
            ts = time.time()
            key  = "WORKER_TERM"
            text = "Manager terminated worker {} (PID {}) on video sequence {}".format(worker_id,worker_pid,in_progress[idx])
            write_to_log(log_file,(ts,key,text))
            
        
            # update progress tracking 
            available[worker_id] = 1
            del in_progress[worker_id]
        
        
            
        
        
        
    print("Finished all video sequences")
    for key in all_workers:
        all_workers[key].terminate()
        all_workers[key].join()
        
    # end log message
    ts = time.time()
    key = "END_PROC_SESSION"
    message = "Ended processing session."
    write_to_log(log_file,(ts,key,message))
        
                
