import torch
import numpy as np
import torch.multiprocessing as mp
import os,sys
import queue
import time
import psutil
import pynvml
import numpy as np

from track_sequence import track_sequence,im_to_vid

import argparse

pynvml.nvmlInit()

sys.path.insert(0,"I24-video-ingest")
from utilities import get_recording_params, find_files

def get_recordings(ingest_session_path):
    params = get_recording_params(ingest_session_path,verbose = False)
    file_list = find_files(params[0],params[1],params[2],drop_last_file = True,verbose = False)
    
    # print("INGEST SESSION PATH: {}".format(ingest_session_path))
    # print(file_list)
    
    keepers = [item[1].split(".mp4")[0] for item in file_list]
    # recording_names = []
    # last_recording_num = {}
    # for item in os.listdir(os.path.join(ingest_session_path,"recording")):        
    #     recording_names.append(item.split(".mp4")[0])
        
    # # remove all recordings that are currently being written to
    # keepers = []
    # for item in recording_names:
    #     for other_item in recording_names:
    #         camera1 = item.split("_")[1]
    #         camera2 = other_item.split("_")[1]
    #         if camera1 == camera2:
    #             num1 = int(item.split("_")[2].split(".mp4")[0])
    #             num2 = int(other_item.split("_")[2].split(".mp4")[0])
    #             if num1 < num2:
    #                 keepers.append(item) # there exists a recording with greater number than item for that camera
        
    return keepers
    
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

def write_to_log(log_file,message,show = False):
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
        if message[1] not in ["SYS"]:
            print(line[:-2])
    with open (log_file,"a+") as f:
        f.writelines([line])
        
def log_system(log_file,process_pids = None):
    """
    Logs system utilization metrics to log file
    """
    # log cpu util 
    cpu_util = psutil.cpu_percent()
    cpu_util_ind = psutil.cpu_percent(percpu = True)
    ts = time.time()
    key = "INFO"
    message = "CPU util: {}% -- Individual utils 1-24: {}".format(cpu_util,cpu_util_ind[:24])
    write_to_log(log_file,(ts,key,message))
    message = "CPU util: {}% -- Individual utils 25-48: {}".format(cpu_util,cpu_util_ind[24:])
    write_to_log(log_file,(ts,key,message))
    
    # log GPU util and memory
    try:
        deviceCount = pynvml.nvmlDeviceGetCount()
        for idx in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            board_num = pynvml.nvmlDeviceGetBoardId(handle)
            name = "GPU {}: {}  (ID {})".format(idx,pynvml.nvmlDeviceGetName(handle).decode("utf-8"),board_num)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            fan_util = pynvml.nvmlDeviceGetFanSpeed(handle)
            pcie_counter = pynvml.nvmlDeviceGetPcieReplayCounter(handle)
            pcie_util = pynvml.nvmlDeviceGetPcieThroughput(handle,pcie_counter)
            gpu_util = util.gpu
            mem_util = util.memory
            
            message = "{}: Kernel:{}%  Mem:{}% Fan:{}% PCIe: {}MB/s".format(name,gpu_util,mem_util,fan_util,round(pcie_util/1000,1))
            ts = time.time()
            key = "INFO"
            write_to_log(log_file,(ts,key,message))
            
    except pynvml.NVMLError as error:
        print(error)
    
    # log memory util
    mem_util = psutil.virtual_memory()
    used = round(mem_util.used / 1e+9,2)
    total = round(mem_util.total / 1e+9,2)
    ts = time.time()
    key = "INFO"
    message = "Memory util: {}%  ({}/{}GB)".format(round(used/total*100,2),used,total)
    write_to_log(log_file,(ts,key,message))
    
    pid_statuses = []
    warning = False
    if process_pids is not None:
        for key in process_pids:
            pid = process_pids[key]
            
            try:
                os.kill(pid,0)
                RUNNING = "running"
            except OSError:
                RUNNING = "stopped"
                warning = True
            
            pid_statuses.append("{} ({}): {}".format(key,pid,RUNNING))
    
        ts = time.time()
        key = "INFO"
        if warning:
            key = "WARNING"
        write_to_log(log_file,(ts,key,pid_statuses))
    
            
    
    
    last_log_time = time.time()    
    return last_log_time        
    

if __name__ == "__main__":
    
    #add argparse block here so we can optionally run from command line
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("Ingest_session_directory", help= '<Required> string, Path to ingest session main directory',type = str)
        parser.add_argument("Configuration_file", help= '<Required> string, Path to configuration file',type = str)
        parser.add_argument("--verbose", help="bool, Show or suppress log messages in terminal", action = "store_true")

        args = parser.parse_args()
        ingest_session_path = args.Ingest_session_directory
        config_file = args.Configuration_file
        if args.verbose:
            VERBOSE = True
        else:
            VERBOSE = False
    except:
        
        print("Using default path instead")
        ingest_session_path = "/home/worklab/Data/cv/video/ingest_session_00011"
        ingest_session_path = "/home/worklab/Data/cv/video/5_min_18_cam_October_2020/ingest_session_00005"
        config_file = "/home/worklab/Documents/derek/I24-video-processing/config/lambda_quad.config"
        VERBOSE = True
    log_rate = 5
    last_log_time = 0
    
    
    # create directory for outputs if needed
    if not os.path.exists(os.path.join(ingest_session_path,"tracking_outputs")):
        os.mkdir(os.path.join(ingest_session_path,"tracking_outputs"))
    
    # define unique log file for processing session
    log_subidx = 0
    log_idx = 0
    while True:
        log_file = os.path.join(ingest_session_path,"logs","cv_tracking_manager_{}_{}.log".format(str(log_idx).zfill(3),log_subidx))
        if os.path.exists(log_file):
            log_idx += 1
        else:
            break
        
        
    write_to_log(log_file,(time.time(),"INFO","STARTED PROCESSING SESSION."),show = VERBOSE)


    # get GPU list
    g = torch.cuda.device_count()

    # availability monitor
    available = np.ones(g)
    in_progress = {}
            
    # create shared queue
    manager = mp.Manager()
    ctx = mp.get_context('spawn')
    com_queue = ctx.Queue()
    all_workers = {}
    process_pids = {"manager":manager._process.ident}
    DONE = False
    
    time_of_last_message = {}
        
    while not DONE:        
        
        try:
            try:
                for idx in range(g):
                    # initially, start gpu_count processes
                    
                    if available[idx] == 1:
                        in_prog = get_in_progress(in_progress)
                        recordings = get_recordings(ingest_session_path)
                        done = get_outputs(ingest_session_path)
                                          
                        if len(in_prog) == 0 and len(recordings) <= len(done):
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
                            
                            input_file_dir = get_recording_params(ingest_session_path)[0][0]
                            input_file = os.path.join(input_file_dir,avail_recording+".mp4")
                            # change to use Will's utilities
                            
                            output_directory = os.path.join(ingest_session_path,"tracking_outputs")
                            camera_id = input_file.split("/")[-1].split("_")[1].upper()
                            
                            args = [input_file, output_directory, config_file,log_file]
                            kwargs = {"worker_id":idx, "com_queue":com_queue,"com_rate": log_rate,"config":camera_id}
                            
                            worker = ctx.Process(target=track_sequence,args = args, kwargs=kwargs)
                            all_workers[idx] = (worker)
                            all_workers[idx].start()
                            
                            # write log message
                            ts = time.time()
                            key  = "DEBUG"
                            text = "Manager started worker {} (PID {}) on video sequence {}".format(idx,all_workers[idx].pid,in_progress[idx])
                            write_to_log(log_file,(ts,key,text),show = VERBOSE)
            except:
                ts = time.time()
                key  = "ERROR"
                text = "Manager had error starting a new process running"
                write_to_log(log_file,(ts,key,text),show = VERBOSE)
                raise KeyboardInterrupt
                        
            
            try:
                # periodically, write device status to log file
                if time.time() - last_log_time > log_rate:
                    last_log_time = log_system(log_file,process_pids)
            except:
                ts = time.time()
                key  = "ERROR"
                text = "Manager had error logging system info"
                write_to_log(log_file,(ts,key,text),show = VERBOSE)
                raise KeyboardInterrupt
            
            
            # monitor queue for messages 
            try:
               message = com_queue.get(timeout = 0)            
            except queue.Empty:
                continue
            
            # strip PID from message and use to update process_pids
            try:
                if "Loader " in message[2]:
                    pid = int(message[2].split("PID ")[1].split(")")[0])
                    id = int(message[2].split("Loader ")[1].split(" ")[0])
                    process_pids["loader {}".format(id)] = pid
                elif "Worker " in message[2]: 
                    pid = int(message[2].split("PID ")[1].split(")")[0])
                    id = int(message[2].split("Worker ")[1].split(" ")[0])
                    process_pids["worker {}".format(id)] = pid
            except:
                ts = time.time()
                key  = "ERROR"
                text = "Manager error parsing PID and ID from message: {}".format(message[2])
                write_to_log(log_file,(ts,key,text),show = VERBOSE)
                raise KeyboardInterrupt
            
            try:
                # write message to log file
                worker_id = message[3]
                message = message[:3]
                write_to_log(log_file,message,show = VERBOSE)
            
            except:
                ts = time.time()
                key  = "ERROR"
                text = "Manager error writing message to log file."
                write_to_log(log_file,(ts,key,text),show = VERBOSE)
                raise KeyboardInterrupt
            
            try:
                time_of_last_message[worker_id] = time.time()
                
                # if message is a finished task, update manager
                key = message[1]
                if key == "WORKER_END":
                    worker_pid = all_workers[worker_id].pid
                    
                    all_workers[worker_id].terminate()
                    all_workers[worker_id].join()
                    del all_workers[worker_id]
                    
                    # write log message
                    ts = time.time()
                    key  = "DEBUG"
                    text = "Manager terminated worker {} (PID {}) on video sequence {}".format(worker_id,worker_pid,in_progress[worker_id])
                    write_to_log(log_file,(ts,key,text),show = VERBOSE)
                    
                
                    # update progress tracking 
                    available[worker_id] = 1
                    del in_progress[worker_id]
            except:
                ts = time.time()
                key  = "ERROR"
                text = "Manager error shutting down finished process"
                write_to_log(log_file,(ts,key,text),show = VERBOSE)
                raise KeyboardInterrupt
            
            try:
                # check for unresponsive processes (no output messages in last 60 seconds, and restart these)
                for worker_id in time_of_last_message:
                    if time.time() - time_of_last_message[worker_id] > 60:
                        # kill process
                        worker_pid = all_workers[worker_id].pid
                        all_workers[worker_id].terminate()
                        all_workers[worker_id].join()
                        del all_workers[worker_id]
                
                        # write log message
                        ts = time.time()
                        key  = "WARNING"
                        text = "Manager terminated unresponsive worker {} (PID {}) on video sequence {}".format(worker_id,worker_pid,in_progress[worker_id])
                        write_to_log(log_file,(ts,key,text),show = VERBOSE)
                        
                                    
                        # update progress tracking 
                        available[worker_id] = 1
                        del in_progress[worker_id]
            except:
                ts = time.time()
                key  = "ERROR"
                text = "Manager error terminating unresponsive process"
                write_to_log(log_file,(ts,key,text),show = VERBOSE)
                raise KeyboardInterrupt
            
            try:
                # make new log file if necessary
                if os.stat(log_file).st_size > 1e+07: # slice into 10 MB log files
                    log_subidx += 1
                    log_file = os.path.join(ingest_session_path,"logs","cv_tracking_manager_{}_{}.log".format(str(log_idx).zfill(3),log_subidx))
            except:
                ts = time.time()
                key  = "ERROR"
                text = "Manager error creating new log file"
                write_to_log(log_file,(ts,key,text),show = VERBOSE)
                raise KeyboardInterrupt


        except KeyboardInterrupt:
            # interrupt log message
            ts = time.time()
            key = "WARNING"
            message = "Keyboard Interrupt error caught. Shutting down worker processes now."
            write_to_log(log_file,(ts,key,message),show = VERBOSE)
            
            # terminate all worker processes (they will in turn terminate their daemon loaders)
            for worker in all_workers:
                all_workers[worker].terminate()
                all_workers[worker].join()
            
            # interrupt log message
            ts = time.time()
            key = "DEBUG"
            message = "All worker processes have been terminated."
            write_to_log(log_file,(ts,key,message),show = VERBOSE)
                        
            break # get out of processing main loop

    
    if DONE:    
        print("Finished all video sequences")
        for key in all_workers:
            all_workers[key].terminate()
            all_workers[key].join()
        
    # end log message
    ts = time.time()
    key = "INFO"
    message = "ENDED PROCESSING SESSION."
    write_to_log(log_file,(ts,key,message),show = VERBOSE)
        
                
