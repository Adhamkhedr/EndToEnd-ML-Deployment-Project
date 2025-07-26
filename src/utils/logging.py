#we passed a logger name (usually using __name__) to the get_logger function. 
#Inside the function, we define the name of the folder where all log files will be stored
# which is "logs". We then create that folder using os.makedirs, 
# but it only gets created if it doesn't already exist. 
# After that, we generate a log file name based on the current date (e.g., 2025-07-26.log). 
# Finally, we join this file name with the folder name using os.path.join, 
# which gives us the full path to the log file inside the logs/ folder.

import logging
import os    
from datetime import datetime   

def get_logger(name: str):   
    # will automatically set name to be the name of the 
    # current file, so if we're in if train.py file , then name == "train". So as if im saying 
    # hello this message is coming from ... file  

    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

#create the full path for the log file we'll write to 
    log_file = os.path.join(logs_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log") 
#os.path.join("logs", "2025-07-26.log")   >  logs/2025-07-26.log
#Create or write to a file called 2025-07-26.log inside the logs/ folder
   
    logging.basicConfig(   #setting up the rules for how logging should behave in the program.
        filename=log_file,               # Send logs to the log file
        level=logging.INFO,              # Log messages of INFO level or higher
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )   #how the log messages should look inside the file.
#2025-07-26 22:53:47,123 - train - INFO - Training started
 
    return logging.getLogger(name)
