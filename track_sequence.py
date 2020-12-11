import torch
import time
import sys,os
import _pickle as pickle
# tracker

# CNNs


sys.path.insert(0,os.path.join(os.getcwd(),"localization-based-tracking"))
from tracker import Localization_Tracker

detector_path = os.path.join(os.getcwd(),"localization-based-tracking","models","pytorch_retinanet_detector")
sys.path.insert(0,detector_path)
from models.pytorch_retinanet_detector.retinanet.model import resnet50 

localizer_path = os.path.join(os.getcwd(),"localization-based-tracking","models","pytorch_retinanet_localizer")
sys.path.insert(0,localizer_path)
from models.pytorch_retinanet_localizer.retinanet.model import resnet34


def parse_config_file(config_file):
    all_blocks = []
    current_block = None
    with open(config_file, 'r') as f:
        for line in f:
            # ignore empty lines and comment lines
            if line is None or len(line.strip()) == 0 or line[0] == '#':
                continue
            strip_line = line.strip()
            if len(strip_line) > 2 and strip_line[:2] == '__' and strip_line[-2:] == '__':
                # this is a configuration block line
                # first check if this is the first one or not
                if current_block is not None:
                    all_blocks.append(current_block)
                current_block = {}
                current_block["name"] = str(strip_line[2:-2])

            elif '==' in strip_line:
                pkey, pval = strip_line.split('==')
                pkey = pkey.strip()
                pval = pval.strip()
                
                # parse out non-string values
                try:
                    pval = int(pval)
                except ValueError:
                    try:    
                        pval = float(pval)    
                    except ValueError:
                        pass 
                if pval == "None":
                    pval = None
                if pval == "True":
                    pval = True
                if pval == "False":
                    pval = False
                    
                current_block[pkey] = pval
                
            else:
                raise AttributeError("""Got a line in the configuration file that isn't a block header nor a 
                key=value.\nLine: {}""".format(strip_line))
        # add the last block of the file (if it's non-empty)
        all_blocks.append(current_block)
        
    return all_blocks

def track_sequence(input_file, 
                   output_directory,
                   config_file, 
                   log_file,
                   com_queue = None,
                   config = "DEFAULT",
                   worker_id = 0,
                   com_rate = 15):
    """
    Tracks a video sequence according to the parameters specified in config_file
    
    Parameters
    ----------
    config_file: string
        Path to configuration file with parameter settings for tracker.
    input file: string
        Path to video sequence to be tracked
    output_file: string
        Path to directory where tracking outputs should be written
    log_file : string
        Path to log file 
    com_queue: multiprocessing Queue, (optional)
        Queue for communicating with manager process in a multiprocess scenario
    config : string, (optional)
        Specifies which camera configuration within config file should be used
    worker_id int
        Specifies worker ID assigned by manager process if any
    device : torch.device (optional)
        Specifies which GPU should be used
    """
    
    # write to queue that worker has started
    if com_queue is not None:
        start = time.time()
        key = "DEBUG"
        message = "Worker {} (PID {}) is executing".format(worker_id,os.getpid())
        com_queue.put((start,key,message,worker_id))
    
    # 1. parse config file
    configs = parse_config_file(config_file)
    for configuration in configs:
        if configuration["name"] == config:
            break
    
    # 2. initialize tracker with parsed parameters
    
    # enable CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device(worker_id if use_cuda else "cpu")    
    
    det_step = configuration["det_step"]
    skip_step = configuration["skip_step"]
    
    # load kf_params
    with open(configuration["kf_parameter_path"] ,"rb") as f:
        kf_params = pickle.load(f)
        # these adjustments make the filter a bit less laggy
        kf_params["R"] /= 20
        kf_params["R2"] /= 500 
    
    # load class_dict
    with open(configuration["class_dict_path"] ,"rb") as f:
        class_dict = pickle.load(f)
    
    # load detector
    det_cp = configuration["detector_parameters"]
    detector = resnet50(configuration["num_classes"],device_id = worker_id)
    detector.load_state_dict(torch.load(det_cp))
    detector = detector.to(device)
    detector.eval()
    detector.training = False  
    detector.freeze_bn()
    
    # load localizer
    if configuration["localize"]:
        loc_cp = configuration["localizer_parameters"]
        localizer = resnet34(configuration["num_classes"],device_id = worker_id)
        localizer.load_state_dict(torch.load(loc_cp))
        localizer = localizer.to(device)
        localizer.eval()
        localizer.training = False   
    else:
        localizer = None
    
    if com_queue is not None:
        d1 = localizer.regressionModel.conv1.weight.device
        d2 = detector.regressionModel.conv1.weight.device
        ts = time.time()
        key = "DEBUG"
        message = "Worker {} (PID {}): Localizer on device {}. Detector on device {}".format(worker_id,os.getpid(),d1,d2)
        com_queue.put((ts,key,message,worker_id))
    
    # load other params
    init_frames= configuration["init_frames"]
    fsld_max = configuration["fsld_max"]
    matching_cutoff = configuration["matching_cutoff"]
    ber = configuration["box_expansion_ratio"]
    iou_cutoff = configuration["iou_cutoff"]
    det_conf_cutoff = configuration["det_conf_cutoff"]
    SHOW = configuration["show_tracking"]
    output_video_path = configuration["output_video_path"]
    checksum_path = configuration["checksum_path"]
    geom_path = configuration["geom_path"]

    # make a new output directory if not yet initialized
    try:
        os.mkdir(output_directory)
    except FileExistsError:
        pass
    
    tracker = Localization_Tracker(input_file,
                                   detector,
                                   localizer,
                                   kf_params,
                                   class_dict,
                                   device_id = worker_id,
                                   det_step = det_step,
                                   init_frames = init_frames,
                                   ber = ber,
                                   det_conf_cutoff= det_conf_cutoff,
                                   fsld_max = fsld_max,
                                   matching_cutoff = matching_cutoff,
                                   iou_cutoff = iou_cutoff,
                                   PLOT= SHOW,
                                   OUT = output_video_path,
                                   skip_step = skip_step,
                                   checksum_path = checksum_path,
                                   geom_path = geom_path,
                                   output_dir = output_directory,
                                   com_queue = com_queue,
                                   com_rate = com_rate)
    
    #3. track and write output
    tracker.track()
    tracker.write_results_csv()
     
    if com_queue is not None:
       # write to queue that worker has finished
        end = time.time()
        key = "WORKER_END"
        message = "Worker {} (PID {}) is done executing".format(worker_id,os.getpid())
        com_queue.put((end,key,message,worker_id))

    
    
if __name__ == "__main__":
    input_file = "/home/worklab/Documents/derek/I24-video-processing//localization-based-tracking/demo/record_p1c0_00000.mp4"
    config_file = "/home/worklab/Documents/derek/I24-video-processing/config/tracker_setup.config"
    output_directory = "/home/worklab/Data/cv/video/ingest_session_00011/tracking_outputs"
    log_file = None
    
    track_sequence(input_file,output_directory,config_file,log_file)
    
    